"""Read ``.txt``, ``.md`` and ``.pdf`` files from disk into documents."""

import logging
from collections.abc import Iterator
from pathlib import Path

from grounded_rag.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".pdf"})


def load_path(path: Path) -> Iterator[Document]:
    """Yield a document for ``path`` if it is a file, or for every supported file
    below it if it is a directory. Document ids are paths relative to the root, so
    they stay stable when the same folder is ingested again."""
    path = path.resolve()
    if path.is_file():
        root, candidates = path.parent, [path]
    elif path.is_dir():
        root, candidates = path, sorted(p for p in path.rglob("*") if p.is_file())
    else:
        raise FileNotFoundError(path)

    for candidate in candidates:
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            document = load_file(candidate, root)
        except Exception:
            # Parsers fail in many ways on malformed files; one bad file should
            # not abort the whole run.
            logger.warning("Skipping unreadable file %s", candidate, exc_info=True)
            continue
        if document is None:
            logger.warning("Skipping %s: no extractable text", candidate)
            continue
        yield document


def load_file(path: Path, root: Path) -> Document | None:
    """Load one file, or return ``None`` when it contains no text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, title = _read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _markdown_title(text) if suffix == ".md" else ""
    text = text.strip()
    if not text:
        return None
    relative = path.relative_to(root).as_posix()
    return Document(doc_id=relative, text=text, title=title, source=relative)


def _read_pdf(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    title = (reader.metadata.title or "") if reader.metadata else ""
    return text, title.strip()


def _markdown_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""
