"""Command line interface: ingest documents, query them, and run the API server."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from grounded_rag.config import get_settings
from grounded_rag.documents import Hit
from grounded_rag.generate import GenerationError
from grounded_rag.loaders import load_path
from grounded_rag.pipeline import IndexStats
from grounded_rag.services import Services, build_services

_PREVIEW_CHARS = 200


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Retrieved text can contain characters the console encoding cannot represent.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
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

    return parser


def _add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction)
    parser.add_argument("--collection", help="collection to read from (default: from settings)")


def _ingest(args: argparse.Namespace) -> int:
    services = build_services(get_settings(), collection=args.collection)

    def report(stats: IndexStats) -> None:
        print(f"\r{stats.documents} documents, {stats.chunks} chunks", end="", flush=True)

    try:
        stats = services.indexer.index(load_path(args.path), on_progress=report)
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
