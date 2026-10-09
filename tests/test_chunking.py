import pytest

from grounded_rag.chunking import _SEPARATORS, _atomize, chunk_document, split_text
from grounded_rag.documents import Document

PARAGRAPH = (
    "Hybrid search combines two signals. Dense vectors capture meaning. "
    "Sparse vectors capture exact terms. Fusion merges both rankings."
)
TEXT = "\n\n".join([PARAGRAPH] * 6)


def test_short_text_is_one_chunk():
    assert split_text("A short note.", chunk_size=100) == ["A short note."]


def test_empty_and_whitespace_text_give_no_chunks():
    assert split_text("", chunk_size=100) == []
    assert split_text("  \n\n  ", chunk_size=100) == []


@pytest.mark.parametrize("chunk_size", [40, 80, 200, 500])
@pytest.mark.parametrize("overlap", [0, 20])
def test_chunks_never_exceed_chunk_size(chunk_size, overlap):
    chunks = split_text(TEXT, chunk_size=chunk_size, overlap=overlap)
    assert chunks
    assert max(len(chunk) for chunk in chunks) <= chunk_size


def test_pieces_reassemble_to_the_original_text():
    assert "".join(_atomize(TEXT, 50, _SEPARATORS)) == TEXT


def test_without_overlap_no_text_is_lost_or_repeated():
    chunks = split_text(TEXT, chunk_size=120, overlap=0)
    assert " ".join(chunks).split() == TEXT.split()


def test_overlap_repeats_the_end_of_the_previous_chunk():
    sentences = [f"Sentence number {i} ends here. " for i in range(12)]
    chunks = split_text("".join(sentences), chunk_size=100, overlap=40)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = current.split(". ")[0]
        assert first_sentence in previous


def test_overlap_starts_on_a_sentence_boundary():
    sentences = [f"Sentence number {i} ends here. " for i in range(12)]
    chunks = split_text("".join(sentences), chunk_size=100, overlap=40)
    assert all(chunk.startswith("Sentence number") for chunk in chunks)


def test_long_token_is_hard_cut():
    chunks = split_text("x" * 250, chunk_size=100)
    assert [len(chunk) for chunk in chunks] == [100, 100, 50]


def test_paragraph_boundaries_are_preferred():
    text = "First paragraph here.\n\nSecond paragraph here."
    assert split_text(text, chunk_size=30) == ["First paragraph here.", "Second paragraph here."]


@pytest.mark.parametrize(("chunk_size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size, overlap=overlap)


def test_chunk_ids_are_deterministic_and_unique():
    document = Document(doc_id="guide.md", text=TEXT, title="Guide", source="guide.md")
    first = chunk_document(document, chunk_size=120, overlap=20)
    second = chunk_document(document, chunk_size=120, overlap=20)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert [chunk.index for chunk in first] == list(range(len(first)))


def test_chunk_ids_differ_between_documents():
    a = chunk_document(Document(doc_id="a", text="same text"), chunk_size=100)
    b = chunk_document(Document(doc_id="b", text="same text"), chunk_size=100)
    assert a[0].chunk_id != b[0].chunk_id


def test_embedding_text_includes_the_title():
    chunk = chunk_document(Document(doc_id="a", text="Body.", title="Title"), chunk_size=100)[0]
    assert chunk.embedding_text == "Title\nBody."
    untitled = chunk_document(Document(doc_id="b", text="Body."), chunk_size=100)[0]
    assert untitled.embedding_text == "Body."
