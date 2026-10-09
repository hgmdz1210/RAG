import pytest

from rag.auth import User
from rag.db import STATUS_FAILED, STATUS_READY
from rag.generate import Answer
from rag.library import DuplicateDocument, UploadRejected
from rag.storage import StorageError

ALICE = User(id="alice")
BOB = User(id="bob")
REPORT = b"Quarterly revenue grew because subscriptions renewed at a higher rate."


@pytest.fixture
def library(services):
    return services.library


def test_add_indexes_the_file_and_marks_it_ready(library, services):
    record = library.add(ALICE, "report.txt", REPORT)

    assert record.status == STATUS_READY
    assert record.chunks == 1
    assert record.filename == "report.txt"
    assert record.size_bytes == len(REPORT)
    assert library.storage.files == {f"alice/{record.id}.txt": REPORT}
    hits = services.retriever.retrieve("quarterly revenue", mode="hybrid", owner="alice")
    assert [hit.chunk.doc_id for hit in hits] == [record.id]
    assert hits[0].chunk.source == "report.txt"


def test_documents_are_listed_newest_first_and_only_for_their_owner(library):
    first = library.add(ALICE, "first.txt", b"The first note is about onboarding.")
    second = library.add(ALICE, "second.txt", b"The second note is about billing.")
    library.add(BOB, "bob.txt", b"Bob keeps his own notes here.")

    assert [record.id for record in library.list(ALICE)] == [second.id, first.id]
    assert [record.filename for record in library.list(BOB)] == ["bob.txt"]


def test_the_same_content_twice_is_a_duplicate_for_that_user_only(library):
    original = library.add(ALICE, "report.txt", REPORT)

    with pytest.raises(DuplicateDocument) as raised:
        library.add(ALICE, "copy-of-report.txt", REPORT)

    assert raised.value.existing.id == original.id
    assert len(library.list(ALICE)) == 1
    assert library.add(BOB, "report.txt", REPORT).status == STATUS_READY


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        pytest.param("tool.exe", b"MZ", "Unsupported file type", id="wrong-type"),
        pytest.param("empty.txt", b"", "empty", id="empty"),
        pytest.param("blank.txt", b"   \n  ", "No text", id="blank"),
        pytest.param("big.txt", b"x" * 1_000_001, "larger than", id="too-large"),
        pytest.param("broken.pdf", b"not a pdf at all", "could not be read", id="corrupt-pdf"),
    ],
)
def test_unusable_files_are_rejected_without_leaving_anything_behind(
    library, services, filename, data, message
):
    with pytest.raises(UploadRejected, match=message):
        library.add(ALICE, filename, data)

    assert library.list(ALICE) == []
    assert library.storage.files == {}
    assert services.store.count() == 0


def test_only_the_base_name_of_a_client_supplied_path_is_kept(library):
    record = library.add(ALICE, "..\\..\\secrets/../report.txt", REPORT)

    assert record.filename == "report.txt"
    assert record.storage_path == f"alice/{record.id}.txt"


def test_a_failed_upload_is_recorded_as_failed_and_cleaned_up(library, services):
    library.storage.fail = StorageError("Could not reach file storage.")

    record = library.add(ALICE, "report.txt", REPORT)

    assert record.status == STATUS_FAILED
    assert record.error == "Could not reach file storage."
    assert record.chunks == 0
    assert services.store.count() == 0
    assert [r.status for r in library.list(ALICE)] == [STATUS_FAILED]


def test_delete_removes_the_row_the_file_and_the_chunks(library, services):
    kept = library.add(ALICE, "kept.txt", b"This note stays in the library.")
    removed = library.add(ALICE, "removed.txt", REPORT)

    assert library.delete(ALICE, removed.id) is True

    assert [record.id for record in library.list(ALICE)] == [kept.id]
    assert list(library.storage.files) == [f"alice/{kept.id}.txt"]
    assert services.store.count() == 1


def test_a_user_cannot_delete_another_users_document(library, services):
    record = library.add(ALICE, "report.txt", REPORT)

    assert library.delete(BOB, record.id) is False

    assert len(library.list(ALICE)) == 1
    assert services.store.count() == 1


def test_deleting_an_unknown_document_reports_false(library):
    assert library.delete(ALICE, "no-such-id") is False


def test_history_keeps_answers_per_user_newest_first(library, services):
    record = library.add(ALICE, "report.txt", REPORT)
    hits = services.retriever.retrieve("quarterly revenue", owner="alice")

    library.record_answer(ALICE, "first?", Answer(text="One [1].", cited=[1], model="m"), hits)
    library.record_answer(ALICE, "second?", Answer(text="No.", answerable=False), hits)
    library.record_answer(BOB, "bob's question", Answer(text="Bob's answer."), [])

    history = library.history(ALICE)
    assert [message.question for message in history] == ["second?", "first?"]
    assert history[1].sources == [
        {
            "number": 1,
            "doc_id": record.id,
            "title": "",
            "source": "report.txt",
            "cited": True,
        }
    ]
    assert history[0].answerable is False
    assert history[0].sources[0]["cited"] is False
    assert [message.question for message in library.history(BOB)] == ["bob's question"]
    assert len(library.history(ALICE, limit=1)) == 1
