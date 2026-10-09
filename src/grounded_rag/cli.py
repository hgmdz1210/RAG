"""Command line interface: ingest documents, query them, serve the API, run the benchmark."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from grounded_rag.config import get_settings
from grounded_rag.documents import Hit
from grounded_rag.eval import beir, runner
from grounded_rag.generate import GenerationError
from grounded_rag.loaders import load_path
from grounded_rag.pipeline import IndexStats
from grounded_rag.services import Services, build_services

_PREVIEW_CHARS = 200


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # One line per HTTP request to Qdrant or Anthropic would drown the real output.
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Reports and retrieved text contain non-ASCII characters. On Windows a piped
    # stream defaults to a legacy code page that mangles them, so write UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-rag", description=__doc__)
    commands = parser.add_subparsers(required=True, metavar="command")

    ingest = commands.add_parser("ingest", help="index a file or a folder of .pdf/.txt/.md files")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--collection", help="collection to write to (default: from settings)")
    ingest.set_defaults(handler=_ingest)

    search = commands.add_parser("search", help="print the passages retrieved for a query")
    search.add_argument("query")
    _add_retrieval_options(search)
    search.set_defaults(handler=_search)

    ask = commands.add_parser("ask", help="answer a question from the indexed documents")
    ask.add_argument("question")
    _add_retrieval_options(ask)
    ask.set_defaults(handler=_ask)

    serve = commands.add_parser("serve", help="run the HTTP API and the demo page")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)

    evaluate = commands.add_parser(
        "eval", help="benchmark retrieval configurations on a BEIR dataset"
    )
    evaluate.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    evaluate.add_argument(
        "--split",
        default="test",
        help="judgment split to score against; tune on train or dev, report on test",
    )
    evaluate.add_argument(
        "--configs",
        default=",".join(runner.CONFIGS),
        help="comma-separated configurations; the first is the baseline",
    )
    evaluate.add_argument("--limit", type=int, help="evaluate only the first N queries")
    evaluate.add_argument("--k", type=int, default=10, help="cutoff for the metrics")
    evaluate.add_argument(
        "--depth",
        type=int,
        default=40,
        help="chunks retrieved per query before collapsing them to documents",
    )
    evaluate.add_argument("--reindex", action="store_true", help="rebuild the collection first")
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/beir"))
    evaluate.add_argument("--out", type=Path, default=Path("results"))
    evaluate.add_argument("--collection", help="collection name (default: beir_<dataset>)")
    evaluate.set_defaults(handler=_eval)

    return parser


def _add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction)
    parser.add_argument("--collection", help="collection to read from (default: from settings)")


def _ingest(args: argparse.Namespace) -> int:
    services = build_services(get_settings(), collection=args.collection)
    try:
        stats = services.indexer.index(load_path(args.path), on_progress=_report_indexing)
    except FileNotFoundError:
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 2
    if not stats.documents:
        print(f"No readable .pdf, .txt or .md files under {args.path}.", file=sys.stderr)
        return 1
    print(
        f"\rIndexed {stats.documents} documents as {stats.chunks} chunks "
        f"into '{services.store.collection}'."
    )
    return 0


def _search(args: argparse.Namespace) -> int:
    services = build_services(get_settings(), collection=args.collection)
    hits = _retrieve(services, args.query, args)
    if not hits:
        print("No passages found.")
        return 1
    _print_hits(hits)
    return 0


def _ask(args: argparse.Namespace) -> int:
    services = build_services(get_settings(), collection=args.collection)
    hits = _retrieve(services, args.question, args)
    try:
        answer = services.generator.answer(args.question, hits)
    except GenerationError as error:
        print(f"Could not generate an answer: {error}", file=sys.stderr)
        return 1
    print(answer.text)
    if hits:
        print("\nSources (* = cited):")
        _print_hits(hits, cited=answer.cited)
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from grounded_rag.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def _eval(args: argparse.Namespace) -> int:
    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in names if name not in runner.CONFIGS]
    if unknown or not names:
        print(
            f"Unknown configuration(s): {', '.join(unknown) or '(none given)'}. "
            f"Choose from: {', '.join(runner.CONFIGS)}.",
            file=sys.stderr,
        )
        return 2

    beir.download(args.dataset, args.data_dir)
    dataset = beir.load(args.dataset, args.data_dir, args.split)
    if args.limit:
        dataset = runner.limit_queries(dataset, args.limit)

    settings = get_settings()
    services = build_services(settings, collection=args.collection or f"beir_{args.dataset}")
    store = services.store
    if args.reindex and store.count():
        store.drop()
    if store.count() == 0:
        print(f"Indexing {len(dataset.corpus)} documents into '{store.collection}'...")
        services.indexer.index(dataset.corpus, on_progress=_report_indexing)
        print()
    else:
        print(
            f"Reusing collection '{store.collection}' ({store.count()} chunks). "
            "Pass --reindex after changing models or chunking."
        )

    runs = []
    for name in names:

        def report(done: int, total: int, name: str = name) -> None:
            if done % 10 == 0 or done == total:
                print(f"\r{name}: {done}/{total} queries", end="", flush=True)

        runs.append(
            runner.run_config(
                services.retriever,
                dataset,
                runner.CONFIGS[name],
                k=args.k,
                depth=args.depth,
                on_progress=report,
            )
        )
        print()

    report_data = runner.build_report(
        dataset,
        runs,
        k=args.k,
        depth=args.depth,
        settings=settings,
        corpus_chunks=store.count(),
        vector_store=_vector_store_label(services),
    )
    markdown = runner.render_markdown(report_data)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = dataset.name if dataset.split == "test" else f"{dataset.name}-{dataset.split}"
    (args.out / f"{stem}.json").write_text(
        json.dumps(report_data, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(f"\n{markdown}\nWrote {args.out / stem}.json and .md")
    return 0


def _vector_store_label(services: Services) -> str:
    if not services.settings.qdrant_url:
        return "Qdrant embedded"
    return f"Qdrant server {services.store.client.info().version}"


def _report_indexing(stats: IndexStats) -> None:
    print(f"\r{stats.documents} documents, {stats.chunks} chunks", end="", flush=True)


def _retrieve(services: Services, query: str, args: argparse.Namespace) -> list[Hit]:
    settings = services.settings
    return services.retriever.retrieve(
        query,
        mode=args.mode or settings.search_mode,
        top_k=args.top_k or settings.top_k,
        candidate_k=settings.candidate_k,
        rerank=settings.rerank if args.rerank is None else args.rerank,
    )


def _print_hits(hits: Sequence[Hit], cited: Sequence[int] = ()) -> None:
    for number, hit in enumerate(hits, start=1):
        marker = "*" if number in cited else " "
        chunk = hit.chunk
        preview = " ".join(chunk.text.split())
        if len(preview) > _PREVIEW_CHARS:
            preview = preview[:_PREVIEW_CHARS].rstrip() + "..."
        print(f"{marker}[{number}] {chunk.doc_id}#{chunk.index}  score={hit.score:.3f}")
        print(f"      {preview}")


if __name__ == "__main__":
    sys.exit(main())
