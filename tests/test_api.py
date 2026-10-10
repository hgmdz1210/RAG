import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from conftest import CORPUS
from rag.api import create_app
from rag.generation.answer import GenerationError


@pytest.fixture
def client(indexed_services) -> TestClient:
    return TestClient(create_app(indexed_services))


def test_health_reports_the_chunk_count(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["collection"] == "test"
    assert body["chunks"] == len(CORPUS)


def test_health_works_before_anything_is_indexed(services):
    response = TestClient(create_app(services)).get("/health")

    assert response.status_code == 200
    assert response.json()["chunks"] == 0


def test_index_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "rag" in response.text


def test_search_uses_the_configured_defaults(client):
    response = client.post("/search", json={"query": "term frequency inverse document frequency"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert body["reranked"] is False
    assert body["hits"][0]["doc_id"] == "bm25.md"
    assert [hit["number"] for hit in body["hits"]] == list(range(1, len(body["hits"]) + 1))


def test_search_honours_request_overrides(client, indexed_services):
    response = client.post(
        "/search", json={"query": "dense vectors", "mode": "sparse", "top_k": 1, "rerank": True}
    )

    body = response.json()
    assert body["mode"] == "sparse"
    assert body["reranked"] is True
    assert len(body["hits"]) == 1
    assert len(indexed_services.retriever.reranker.calls) == 1


def test_reranker_is_not_called_unless_requested(client, indexed_services):
    client.post("/search", json={"query": "dense vectors"})

    assert indexed_services.retriever.reranker.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 500},
        {"query": "x", "mode": "keyword"},
        {},
    ],
)
def test_search_rejects_invalid_requests(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_whitespace_query_is_rejected(client):
    assert client.post("/search", json={"query": "   "}).status_code == 422


def test_ask_returns_the_answer_and_marks_cited_sources(client):
    response = client.post("/ask", json={"question": "How does a cross encoder rescore?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Based on the first passage [1]."
    assert body["model"] == "fake-model"
    assert body["refused"] is False
    assert body["answerable"] is True
    assert [source["cited"] for source in body["sources"]][:2] == [True, False]
    assert body["sources"][0]["doc_id"] == "rerank.md"


@pytest.mark.parametrize(("retryable", "expected_status"), [(True, 503), (False, 502)])
def test_generation_failures_map_to_gateway_errors(
    client, indexed_services, retryable, expected_status
):
    indexed_services.generator.error = GenerationError("upstream failed", retryable=retryable)

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == expected_status
    assert response.json()["detail"] == "upstream failed"


def test_api_key_is_enforced_when_configured(indexed_services):
    indexed_services.settings.api_key = SecretStr("s3cret")
    client = TestClient(create_app(indexed_services))
    payload = {"query": "vectors"}

    assert client.post("/search", json=payload).status_code == 401
    assert client.post("/search", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json={"question": "vectors"}).status_code == 401
    assert client.post("/search", json=payload, headers={"X-API-Key": "s3cret"}).status_code == 200
    # Health stays open so container orchestrators can probe it.
    assert client.get("/health").status_code == 200
