"""Run retrieval configurations over a dataset and report metrics with confidence intervals."""

import dataclasses
import os
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rag.config import SearchMode, Settings
from rag.documents import Hit
from rag.eval import metrics
from rag.eval.beir import BeirDataset
from rag.pipeline import Retriever

_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 0


@dataclass(frozen=True, slots=True)
class EvalConfig:
    name: str
    mode: SearchMode
    rerank: bool = False


CONFIGS: dict[str, EvalConfig] = {
    config.name: config
    for config in (
        EvalConfig("dense", "dense"),
        EvalConfig("bm25", "sparse"),
        EvalConfig("hybrid", "hybrid"),
        EvalConfig("hybrid+rerank", "hybrid", rerank=True),
    )
}


@dataclass(slots=True)
class ConfigRun:
    config: EvalConfig
    # Metric name -> one value per query, in the dataset's query order. Keeping
    # the per-query values is what allows paired comparisons between runs.
    scores: dict[str, list[float]]
    latencies_ms: list[float]


def metric_names(k: int) -> list[str]:
    return [f"ndcg@{k}", f"recall@{k}", f"mrr@{k}", "hit@1"]


def limit_queries(dataset: BeirDataset, count: int) -> BeirDataset:
    """The same dataset restricted to its first ``count`` queries."""
    query_ids = list(dataset.queries)[:count]
    return dataclasses.replace(
        dataset,
        queries={query_id: dataset.queries[query_id] for query_id in query_ids},
        qrels={query_id: dataset.qrels[query_id] for query_id in query_ids},
    )


def rank_documents(hits: Sequence[Hit]) -> list[str]:
    """Collapse chunk hits into a document ranking: each document takes the
    position of its best-ranked chunk. Relevance judgments are per document."""
    return list(dict.fromkeys(hit.chunk.doc_id for hit in hits))


def run_config(
    retriever: Retriever,
    dataset: BeirDataset,
    config: EvalConfig,
    *,
    k: int = 10,
    depth: int = 40,
    on_progress: Callable[[int, int], None] | None = None,
) -> ConfigRun:
    """Evaluate one configuration.

    ``depth`` chunks are retrieved per query (and rescored, when reranking) before
    they are collapsed to documents, so the top ``k`` documents are not starved by
    several chunks of the same document.
    """
    if depth < k:
        raise ValueError("depth must be at least k")

    def retrieve(query: str) -> list[Hit]:
        return retriever.retrieve(
            query, mode=config.mode, top_k=depth, candidate_k=depth, rerank=config.rerank
        )

    scores: dict[str, list[float]] = {name: [] for name in metric_names(k)}
    latencies_ms: list[float] = []
    queries = list(dataset.queries.items())
    if queries:
        # The first call loads models; keep that out of the latency figures.
        retrieve(queries[0][1])

    for done, (query_id, query) in enumerate(queries, start=1):
        started = time.perf_counter()
        hits = retrieve(query)
        latencies_ms.append((time.perf_counter() - started) * 1000)

        ranked = rank_documents(hits)
        gains = dataset.qrels[query_id]
        scores[f"ndcg@{k}"].append(metrics.ndcg_at_k(ranked, gains, k))
        scores[f"recall@{k}"].append(metrics.recall_at_k(ranked, gains, k))
        scores[f"mrr@{k}"].append(metrics.reciprocal_rank(ranked, gains, k))
        scores["hit@1"].append(metrics.hit_at_k(ranked, gains, 1))
        if on_progress:
            on_progress(done, len(queries))

    return ConfigRun(config=config, scores=scores, latencies_ms=latencies_ms)


def build_report(
    dataset: BeirDataset,
    runs: Sequence[ConfigRun],
    *,
    k: int,
    depth: int,
    settings: Settings,
    corpus_chunks: int,
    vector_store: str,
) -> dict[str, Any]:
    """Summarise runs as a JSON-serialisable report.

    Per-query scores are kept in the report so that any two configurations can be
    compared later without running retrieval again.
    """
    headline = f"ndcg@{k}"
    configs = [
        {
            "name": run.config.name,
            "mode": run.config.mode,
            "rerank": run.config.rerank,
            "metrics": {name: _summary(values) for name, values in run.scores.items()},
            "latency_ms": {
                "p50": round(metrics.percentile(run.latencies_ms, 50), 1),
                "p95": round(metrics.percentile(run.latencies_ms, 95), 1),
            },
            "per_query": {
                name: [round(value, 4) for value in values] for name, values in run.scores.items()
            },
        }
        for run in runs
    ]
    comparisons = [
        {
            "config": run.config.name,
            "reference": reference.config.name,
            "metric": headline,
            "delta": _summary(
                [
                    value - other
                    for value, other in zip(
                        run.scores[headline], reference.scores[headline], strict=True
                    )
                ]
            ),
        }
        for run, reference in comparison_pairs(runs)
    ]

    return {
        "dataset": dataset.name,
        "split": dataset.split,
        "queries": len(dataset.queries),
        "query_ids": list(dataset.queries),
        "corpus_documents": len(dataset.corpus),
        "corpus_chunks": corpus_chunks,
        "k": k,
        "depth": depth,
        "baseline": runs[0].config.name,
        "bootstrap": {
            "resamples": _BOOTSTRAP_RESAMPLES,
            "confidence": 0.95,
            "seed": _BOOTSTRAP_SEED,
        },
        "settings": {
            "dense_model": settings.dense_model,
            "dense_query_prefix": settings.dense_query_prefix,
            "sparse_model": settings.sparse_model,
            "rerank_model": settings.rerank_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "rrf_k": settings.rrf_k,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "vector_store": vector_store,
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "configs": configs,
        "comparisons": comparisons,
    }


def comparison_pairs(runs: Sequence[ConfigRun]) -> list[tuple[ConfigRun, ConfigRun]]:
    """Which runs to compare, as (run, reference) pairs.

    Every run is compared with the first one, the baseline. A reranked run is also
    compared with the same search mode without reranking, because that pair isolates
    what the reranker itself contributes.
    """
    baseline = runs[0]
    pairs = [(run, baseline) for run in runs[1:]]
    unreranked = {run.config.mode: run for run in runs if not run.config.rerank}
    for run in runs:
        reference = unreranked.get(run.config.mode)
        if run.config.rerank and reference is not None and reference is not baseline:
            pairs.append((run, reference))
    return pairs


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report as two Markdown tables: absolute metrics, then paired gaps."""
    names = list(report["configs"][0]["metrics"])
    headline = names[0]
    lines = [
        f"Dataset `{report['dataset']}`, {report['split']} split: {report['queries']} queries over "
        f"{report['corpus_documents']:,} documents ({report['corpus_chunks']:,} chunks). "
        "Means with 95% bootstrap confidence intervals.",
        "",
        "| Configuration | " + " | ".join(names) + " | p50 ms | p95 ms |",
        "|---|" + "---|" * (len(names) + 2),
    ]
    for config in report["configs"]:
        cells = [_cell(config["metrics"][name]) for name in names]
        latency = config["latency_ms"]
        lines.append(
            f"| {config['name']} | "
            + " | ".join(cells)
            + f" | {latency['p50']} | {latency['p95']} |"
        )

    if report["comparisons"]:
        lines += [
            "",
            f"Paired differences in {headline}, computed query by query:",
            "",
            f"| Comparison | Δ {headline} | 95% CI |",
            "|---|---|---|",
        ]
        for comparison in report["comparisons"]:
            delta = comparison["delta"]
            low, high = delta["ci95"]
            lines.append(
                f"| {comparison['config']} vs {comparison['reference']} "
                f"| {delta['mean']:+.3f} | {low:+.3f} to {high:+.3f} |"
            )
    return "\n".join(lines) + "\n"


def _summary(values: Sequence[float]) -> dict[str, Any]:
    low, high = metrics.bootstrap_ci(
        values, resamples=_BOOTSTRAP_RESAMPLES, confidence=0.95, seed=_BOOTSTRAP_SEED
    )
    return {"mean": round(metrics.mean(values), 4), "ci95": [round(low, 4), round(high, 4)]}


def _cell(summary: dict[str, Any]) -> str:
    low, high = summary["ci95"]
    return f"{summary['mean']:.3f} ({low:.3f}–{high:.3f})"
