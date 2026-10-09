import pytest

from rag.loaders import load_path


def _minimal_pdf(text: str) -> bytes:
    """Build a one-page PDF containing ``text``, with a correct xref table."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n%s\nendobj\n" % (number, content)
    xref_start = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        body += b"%010d 00000 n \n" % offset
    body += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_start,
    )
    return body


def test_loads_supported_files_with_relative_ids(tmp_path):
    (tmp_path / "notes.txt").write_text("Plain text notes.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "guide.md").write_text("# The Guide\n\nBody text.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    documents = {doc.doc_id: doc for doc in load_path(tmp_path)}

    assert set(documents) == {"notes.txt", "sub/guide.md"}
    assert documents["notes.txt"].title == ""
    assert documents["sub/guide.md"].title == "The Guide"
    assert documents["sub/guide.md"].source == "sub/guide.md"


def test_loads_a_single_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Only this file.", encoding="utf-8")
    (tmp_path / "other.txt").write_text("Not this one.", encoding="utf-8")

    documents = list(load_path(path))

    assert [doc.doc_id for doc in documents] == ["notes.txt"]


def test_extracts_text_from_pdf(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(_minimal_pdf("Hybrid retrieval beats either signal"))

    (document,) = load_path(tmp_path)

    assert document.doc_id == "paper.pdf"
    assert "Hybrid retrieval beats either signal" in document.text


def test_skips_empty_and_corrupt_files(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    (tmp_path / "good.txt").write_text("Readable.", encoding="utf-8")

    documents = list(load_path(tmp_path))

    assert [doc.doc_id for doc in documents] == ["good.txt"]


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(load_path(tmp_path / "does-not-exist"))
