import json
import zipfile

import pytest

from conftest import CORPUS
from rag import cli
from rag.documents import Chunk, Hit
from rag.evaluation import beir, runner

QUERIES = {
    "q1": "term frequency and inverse document frequency",
    "q2": "cross encoder rescores candidate passages",
    "q3": "reciprocal rank fusion merges rankings",
    "unjudged": "a query with no relevance judgments",
}
QRELS = {"q1": {"bm25.md": 1}, "q2": {"rerank.md": 1}, "q3": {"fusion.md": 2}}


def dataset_files(name="tiny") -> dict[str, str]:
    """The shared test corpus as BEIR files: relative path -> content."""
    corpus = [
        json.dumps({"_id": document.doc_id, "title": document.title, "text": document.text})
        for document in CORPUS
    ]
    queries = [json.dumps({"_id": query_id, "text": text}) for query_id, text in QUERIES.items()]
    qrels = ["query-id\tcorpus-id\tscore"]
    qrels += [
        f"{qid}\t{doc_id}\t{grade}" for qid, docs in QRELS.items() for doc_id, grade in docs.items()
    ]
    qrels.append("q1\tchunking.md\t0")  # a judged non-relevant document
    return {
        f"{name}/corpus.jsonl": "\n".join(corpus) + "\n",
        f"{name}/queries.jsonl": "\n".join(queries) + "\n",
        f"{name}/qrels/test.tsv": "\n".join(qrels) + "\n",
    }


def write_dataset(data_dir, name="tiny"):
    for relative, content in dataset_files(name).items():
        path = data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    write_dataset(tmp_path)
    return beir.load("tiny", tmp_path)


def test_load_reads_corpus_queries_and_positive_judgments(dataset):
    assert [document.doc_id for document in dataset.corpus] == [doc.doc_id for doc in CORPUS]
    assert dataset.corpus[0].title == "Qdrant"
    assert dataset.qrels == QRELS
    assert set(dataset.queries) == {"q1", "q2", "q3"}


def test_download_unpacks_the_archive_once(tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(url, path):
        fetched.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as bundle:
            for relative, content in dataset_files().items():
                bundle.writestr(relative, content)

    monkeypatch.setattr(beir, "_fetch", fake_fetch)

    target = beir.download("tiny", tmp_path)
    beir.download("tiny", tmp_path)

    assert fetched == [
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/tiny.zip"
    ]
    assert target == tmp_path / "tiny"
    assert not (tmp_path / "tiny.zip").exists()
    assert beir.load("tiny", tmp_path).qrels == QRELS


def test_archive_entries_outside_the_dataset_folder_are_ignored(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    def fake_fetch(url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as bundle:
            for relative, content in dataset_files().items():
                bundle.writestr(relative, content)
            bundle.writestr("../escaped.txt", "outside the data directory")
            bundle.writestr("tiny/../../escaped-too.txt", "also outside")
            bundle.writestr("other/corpus.jsonl", "a different folder")

    monkeypatch.setattr(beir, "_fetch", fake_fetch)

    beir.download("tiny", data_dir)

    assert (data_dir / "tiny" / "corpus.jsonl").exists()
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "escaped-too.txt").exists()
    assert not (data_dir / "other").exists()


@pytest.mark.parametrize("name", ["../etc", "sci fact", "SciFact", "", "a/b"])
def test_unsafe_dataset_names_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        beir.download(name, tmp_path)


def test_rank_documents_keeps_each_documents_best_position():
    def hit(doc_id, index):
        chunk = Chunk(chunk_id=f"{doc_id}-{index}", doc_id=doc_id, index=index, text="")
        return Hit(chunk=chunk, score=0.0)

    hits = [hit("a", 0), hit("b", 0), hit("a", 1), hit("c", 0), hit("b", 3)]

    assert runner.rank_documents(hits) == ["a", "b", "c"]


def test_run_config_scores_every_query(retriever, dataset):
    seen = []

    run = runner.run_config(
        retriever,
        dataset,
        runner.CONFIGS["hybrid"],
        k=3,
        depth=5,
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert run.scores["ndcg@3"] == [1.0, 1.0, 1.0]
    assert run.scores["hit@1"] == [1.0, 1.0, 1.0]
    assert len(run.latencies_ms) == 3
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_run_config_uses_the_reranker_only_when_configured(retriever, reranker, dataset):
    runner.run_config(retriever, dataset, runner.CONFIGS["dense"], k=3, depth=5)
    assert reranker.calls == []

    runner.run_config(retriever, dataset, runner.CONFIGS["hybrid+rerank"], k=3, depth=5)
    # One warm-up call plus one per query.
    assert len(reranker.calls) == 4


def test_depth_below_k_is_rejected(retriever, dataset):
    with pytest.raises(ValueError):
        runner.run_config(retriever, dataset, runner.CONFIGS["dense"], k=10, depth=5)


def test_limit_queries_keeps_the_first_queries_and_their_judgments(dataset):
    limited = runner.limit_queries(dataset, 2)

    assert list(limited.queries) == ["q1", "q2"]
    assert set(limited.qrels) == {"q1", "q2"}
    assert limited.corpus == dataset.corpus


def test_report_compares_every_run_to_the_first(retriever, dataset, services):
    runs = [
        runner.run_config(retriever, dataset, runner.CONFIGS[name], k=3, depth=5)
        for name in ("dense", "hybrid")
    ]

    report = runner.build_report(
        dataset,
        runs,
        k=3,
        depth=5,
        settings=services.settings,
        corpus_chunks=5,
        vector_store="Qdrant embedded",
    )

    assert report["baseline"] == "dense"
    assert report["queries"] == 3
    assert report["query_ids"] == ["q1", "q2", "q3"]
    dense, hybrid = report["configs"]
    assert dense["metrics"]["ndcg@3"] == {"mean": 1.0, "ci95": [1.0, 1.0]}
    assert dense["per_query"]["ndcg@3"] == [1.0, 1.0, 1.0]
    assert set(hybrid["latency_ms"]) == {"p50", "p95"}
    assert report["comparisons"] == [
        {
            "config": "hybrid",
            "reference": "dense",
            "metric": "ndcg@3",
            "delta": {"mean": 0.0, "ci95": [0.0, 0.0]},
        }
    ]
    json.dumps(report)  # must be serialisable as-is

    markdown = runner.render_markdown(report)
    assert "| dense | 1.000 (1.000–1.000) |" in markdown
    assert "| hybrid vs dense | +0.000 | +0.000 to +0.000 |" in markdown


def _run(name: str, scores: list[float]) -> runner.ConfigRun:
    return runner.ConfigRun(
        config=runner.CONFIGS[name], scores={"ndcg@10": scores}, latencies_ms=[1.0] * len(scores)
    )


def _pair_names(names: list[str]) -> list[tuple[str, str]]:
    runs = [_run(name, [0.5]) for name in names]
    return [(run.config.name, ref.config.name) for run, ref in runner.comparison_pairs(runs)]


def test_a_reranked_run_is_also_compared_with_its_unreranked_counterpart():
    assert _pair_names(["dense", "bm25", "hybrid", "hybrid+rerank"]) == [
        ("bm25", "dense"),
        ("hybrid", "dense"),
        ("hybrid+rerank", "dense"),
        ("hybrid+rerank", "hybrid"),
    ]


def test_the_counterpart_comparison_is_not_repeated_when_it_is_the_baseline():
    assert _pair_names(["hybrid", "hybrid+rerank"]) == [("hybrid+rerank", "hybrid")]


def test_paired_difference_is_the_mean_of_per_query_gaps(dataset, services):
    limited = runner.limit_queries(dataset, 2)
    runs = [_run("hybrid", [0.5, 0.25]), _run("hybrid+rerank", [1.0, 0.25])]

    report = runner.build_report(
        limited,
        runs,
        k=10,
        depth=40,
        settings=services.settings,
        corpus_chunks=5,
        vector_store="Qdrant embedded",
    )

    (comparison,) = report["comparisons"]
    assert comparison["delta"]["mean"] == 0.25
    low, high = comparison["delta"]["ci95"]
    assert 0.0 <= low <= 0.25 <= high <= 0.5


def test_eval_command_indexes_runs_and_writes_reports(tmp_path, services, monkeypatch, capsys):
    data_dir, out_dir = tmp_path / "data", tmp_path / "results"
    write_dataset(data_dir)
    monkeypatch.setattr(cli, "build_services", lambda settings, collection=None: services)
    monkeypatch.setattr(cli.beir, "download", lambda name, data_dir: None)
    arguments = ["eval", "--dataset", "tiny", "--configs", "dense,bm25", "--k", "3", "--depth", "5"]
    arguments += ["--data-dir", str(data_dir), "--out", str(out_dir)]

    assert cli.main(arguments) == 0
    assert "Indexing 5 documents" in capsys.readouterr().out
    report = json.loads((out_dir / "tiny.json").read_text(encoding="utf-8"))
    assert [config["name"] for config in report["configs"]] == ["dense", "bm25"]
    assert report["corpus_chunks"] == 5
    assert (out_dir / "tiny.md").read_text(encoding="utf-8").startswith("Dataset `tiny`")

    # A second run finds the collection populated and does not index again.
    assert cli.main(arguments) == 0
    assert "Reusing collection" in capsys.readouterr().out
    assert services.store.count() == 5


def test_eval_command_rejects_unknown_configurations(capsys):
    assert cli.main(["eval", "--configs", "dense,magic"]) == 2
    assert "magic" in capsys.readouterr().err


def test_eval_command_scores_another_split_into_its_own_report(
    tmp_path, services, monkeypatch, capsys
):
    data_dir, out_dir = tmp_path / "data", tmp_path / "results"
    write_dataset(data_dir)
    (data_dir / "tiny" / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nunjudged\tqdrant.md\t1\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli, "build_services", lambda settings, collection=None: services)
    monkeypatch.setattr(cli.beir, "download", lambda name, data_dir: None)
    arguments = ["eval", "--dataset", "tiny", "--split", "train", "--configs", "dense"]
    arguments += ["--k", "3", "--depth", "5", "--data-dir", str(data_dir), "--out", str(out_dir)]

    assert cli.main(arguments) == 0

    report = json.loads((out_dir / "tiny-train.json").read_text(encoding="utf-8"))
    assert report["split"] == "train"
    assert report["query_ids"] == ["unjudged"]
    assert not (out_dir / "tiny.json").exists()
