import pytest
from fastapi.testclient import TestClient

from conftest import FakeAuth
from rag.api import create_app

ALICE = {"Authorization": "Bearer alice-token"}
BOB = {"Authorization": "Bearer bob-token"}
REPORT = b"Quarterly revenue grew because subscriptions renewed at a higher rate."
MEMO = b"The offsite is planned for the first week of March in Da Nang."


@pytest.fixture
def client(services) -> TestClient:
    """An API with sign-in enabled."""
    services.auth = FakeAuth()
    return TestClient(create_app(services))


def upload(client, headers, name="report.txt", data=REPORT):
    return client.post("/documents", files={"file": (name, data)}, headers=headers)


def test_upload_then_list_then_search_then_delete(client):
    created = upload(client, ALICE)
    assert created.status_code == 201
    document = created.json()
    assert document["filename"] == "report.txt"
    assert document["status"] == "ready"
    assert document["chunks"] == 1

    listed = client.get("/documents", headers=ALICE).json()
    assert [item["id"] for item in listed] == [document["id"]]

    hits = client.post("/search", json={"query": "quarterly revenue"}, headers=ALICE).json()["hits"]
    assert [hit["doc_id"] for hit in hits] == [document["id"]]
    assert hits[0]["source"] == "report.txt"

    assert client.delete(f"/documents/{document['id']}", headers=ALICE).status_code == 204
    assert client.get("/documents", headers=ALICE).json() == []
    assert (
        client.post("/search", json={"query": "quarterly revenue"}, headers=ALICE).json()["hits"]
        == []
    )


def test_users_see_search_and_delete_only_their_own_documents(client):
    alice_doc = upload(client, ALICE).json()
    upload(client, BOB, name="memo.txt", data=MEMO)

    assert [d["filename"] for d in client.get("/documents", headers=BOB).json()] == ["memo.txt"]

    # Bob searches for words that only occur in Alice's document.
    bob_hits = client.post(
        "/search", json={"query": "quarterly revenue subscriptions"}, headers=BOB
    ).json()["hits"]
    assert all(hit["source"] == "memo.txt" for hit in bob_hits)

    assert client.delete(f"/documents/{alice_doc['id']}", headers=BOB).status_code == 404
    assert len(client.get("/documents", headers=ALICE).json()) == 1


def test_ask_uses_only_the_callers_documents_and_is_saved_to_their_history(client):
    document = upload(client, ALICE).json()
    upload(client, BOB, name="memo.txt", data=MEMO)

    response = client.post("/ask", json={"question": "Why did revenue grow?"}, headers=ALICE)

    assert response.status_code == 200
    assert [source["doc_id"] for source in response.json()["sources"]] == [document["id"]]
    history = client.get("/messages", headers=ALICE).json()
    assert [message["question"] for message in history] == ["Why did revenue grow?"]
    assert history[0]["sources"][0]["source"] == "report.txt"
    assert client.get("/messages", headers=BOB).json() == []


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/documents"), ("get", "/messages"), ("post", "/search"), ("post", "/ask")],
)
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer unknown-token"}, {"Authorization": "alice-token"}],
    ids=["no-header", "unknown-token", "not-a-bearer-token"],
)
def test_routes_require_a_valid_session_when_sign_in_is_enabled(client, method, path, headers):
    response = client.request(method, path, json={"query": "x", "question": "x"}, headers=headers)

    assert response.status_code == 401


def test_upload_requires_a_valid_session(client):
    assert upload(client, {}).status_code == 401


def test_health_and_the_demo_page_stay_open(client):
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_a_shared_api_key_does_not_replace_signing_in(client, services):
    from pydantic import SecretStr

    services.settings.api_key = SecretStr("s3cret")

    assert client.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 401


def test_duplicate_upload_is_a_conflict(client):
    upload(client, ALICE)

    again = upload(client, ALICE, name="renamed.txt")

    assert again.status_code == 409
    assert "report.txt" in again.json()["detail"]


def test_unsupported_and_oversized_uploads_are_rejected(client, services):
    assert upload(client, ALICE, name="tool.exe", data=b"MZ").status_code == 422
    assert upload(client, ALICE, name="blank.txt", data=b"  ").status_code == 422

    services.library.max_upload_bytes = 10
    too_large = upload(client, ALICE, name="big.txt", data=b"x" * 11)

    assert too_large.status_code == 413
    assert client.get("/documents", headers=ALICE).json() == []


def test_without_sign_in_there_is_one_local_user(services):
    client = TestClient(create_app(services))

    assert upload(client, {}).status_code == 201
    assert len(client.get("/documents").json()) == 1
    hits = client.post("/search", json={"query": "quarterly revenue"}).json()["hits"]
    assert hits[0]["source"] == "report.txt"
