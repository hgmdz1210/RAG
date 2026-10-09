"""Recursive character chunking with piece-level overlap and deterministic chunk ids."""

import uuid

from grounded_rag.documents import Chunk, Document

# Tried in order: paragraphs, lines, sentences, words.
_SEPARATORS = ("\n\n", "\n", ". ", " ")
_CHUNK_NAMESPACE = uuid.UUID("6f1d2c3a-8b4e-4f7a-9c1d-2e5b7a9d0c11")


def split_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    The text is first broken into pieces at the coarsest separator that keeps every
    piece within ``chunk_size``; pieces are then packed greedily into chunks. Each
    chunk after the first starts with the trailing pieces of the previous one, up to
    ``overlap`` characters. Overlap is taken in whole pieces, so it never starts
    mid-sentence, and a piece larger than ``overlap`` is not repeated.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in _atomize(text, chunk_size, _SEPARATORS):
        if current and length + len(piece) > chunk_size:
            chunks.append("".join(current))
            current = _tail(current, min(overlap, chunk_size - len(piece)))
            length = sum(len(p) for p in current)
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current))
    return [stripped for chunk in chunks if (stripped := chunk.strip())]


def chunk_document(document: Document, chunk_size: int, overlap: int = 0) -> list[Chunk]:
    """Chunk a document. Ids depend only on the document id and chunk position, so
    indexing the same document twice overwrites its chunks instead of duplicating them."""
    return [
        Chunk(
            chunk_id=str(uuid.uuid5(_CHUNK_NAMESPACE, f"{document.doc_id}\x1f{index}")),
            doc_id=document.doc_id,
            index=index,
            text=text,
            title=document.title,
            source=document.source,
        )
        for index, text in enumerate(split_text(document.text, chunk_size, overlap))
    ]


def _atomize(text: str, max_len: int, separators: tuple[str, ...]) -> list[str]:
    """Break ``text`` into pieces no longer than ``max_len``.

    Every piece keeps its trailing separator, so joining the pieces gives back the
    original text exactly.
    """
    if len(text) <= max_len:
        return [text] if text else []
    for position, separator in enumerate(separators):
        if separator not in text:
            continue
        parts = text.split(separator)
        pieces: list[str] = []
        for part_index, part in enumerate(parts):
            piece = part + (separator if part_index < len(parts) - 1 else "")
            pieces.extend(_atomize(piece, max_len, separators[position + 1 :]))
        return pieces
    # A single token longer than the limit: cut it at fixed offsets.
    return [text[start : start + max_len] for start in range(0, len(text), max_len)]


def _tail(pieces: list[str], budget: int) -> list[str]:
    """The longest run of trailing pieces whose total length fits in ``budget``."""
    tail: list[str] = []
    used = 0
    for piece in reversed(pieces):
        if used + len(piece) > budget:
            break
        tail.append(piece)
        used += len(piece)
    tail.reverse()
    return tail
