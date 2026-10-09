"""Download and read retrieval datasets published in the BEIR format.

A dataset is a corpus, a set of queries, and human relevance judgments (qrels)
saying which documents answer which query.
"""

import json
import logging
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from rag.documents import Document

logger = logging.getLogger(__name__)

_ARCHIVE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class BeirDataset:
    name: str
    corpus: list[Document]
    # Query id -> text, restricted to queries with at least one relevant document.
    queries: dict[str, str]
    # Query id -> {document id -> relevance grade}, positive grades only.
    qrels: dict[str, dict[str, int]]
    split: str = "test"


def download(name: str, data_dir: Path) -> Path:
    """Fetch and unpack a dataset into ``data_dir/name`` unless it is already there."""
    _check_name(name)
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    archive = data_dir / f"{name}.zip"
    url = _ARCHIVE_URL.format(name=name)
    logger.info("Downloading %s", url)
    _fetch(url, archive)
    _extract(archive, data_dir, name)
    archive.unlink()
    return target


def load(name: str, data_dir: Path, split: str = "test") -> BeirDataset:
    """Read a downloaded dataset from ``data_dir/name``."""
    _check_name(name)
    _check_name(split)
    directory = data_dir / name

    corpus = [
        Document(
            doc_id=str(row["_id"]),
            text=row.get("text", ""),
            title=row.get("title", ""),
            source=str(row["_id"]),
        )
        for row in _read_jsonl(directory / "corpus.jsonl")
    ]

    qrels: dict[str, dict[str, int]] = {}
    with open(directory / "qrels" / f"{split}.tsv", encoding="utf-8") as lines:
        next(lines)  # header: query-id, corpus-id, score
        for line in lines:
            if not line.strip():
                continue
            query_id, doc_id, grade = line.rstrip("\n").split("\t")
            if int(grade) > 0:
                qrels.setdefault(query_id, {})[doc_id] = int(grade)

    queries = {
        str(row["_id"]): row["text"]
        for row in _read_jsonl(directory / "queries.jsonl")
        if str(row["_id"]) in qrels
    }
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels, split=split)


def _check_name(value: str) -> None:
    # Names end up in URLs and file paths, so only plain slugs are accepted.
    if not _NAME.fullmatch(value):
        raise ValueError(f"invalid dataset or split name: {value!r}")


def _fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "rag"})
    with (
        urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response,
        open(partial, "wb") as output,
    ):
        shutil.copyfileobj(response, output)
    # Rename only after a complete download, so an interrupted run is retried.
    partial.replace(path)


def _extract(archive: Path, data_dir: Path, name: str) -> None:
    """Unpack the archive's ``name/`` folder, ignoring every entry that would land
    anywhere else. Archive entry names are untrusted and may contain ``..``."""
    root = (data_dir / name).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (data_dir / member.filename).resolve()
            if member.is_dir() or not destination.is_relative_to(root):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, open(destination, "wb") as output:
                shutil.copyfileobj(source, output)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]
