import httpx
import pytest

from rag.auth import AuthUnavailable, SupabaseAuthenticator, User
from rag.db import make_engine
from rag.storage import LocalStorage, StorageError, SupabaseStorage


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_authenticator_returns_the_user_for_a_valid_token():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "user-1", "email": "a@example.com"})

    auth = SupabaseAuthenticator("https://p.supabase.co/", "anon", client=_client(handler))

    assert auth.verify("token-1") == User(id="user-1", email="a@example.com")
    assert str(seen[0].url) == "https://p.supabase.co/auth/v1/user"
    assert seen[0].headers["apikey"] == "anon"
    assert seen[0].headers["authorization"] == "Bearer token-1"


def test_authenticator_caches_a_verified_token():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"id": "user-1"})

    auth = SupabaseAuthenticator("https://p.supabase.co", "anon", client=_client(handler))
    auth.verify("token-1")
    auth.verify("token-1")
    auth.verify("token-2")

    assert len(calls) == 2


@pytest.mark.parametrize("status", [401, 403])
def test_authenticator_rejects_an_invalid_token(status):
    auth = SupabaseAuthenticator(
        "https://p.supabase.co", "anon", client=_client(lambda request: httpx.Response(status))
    )

    assert auth.verify("bad") is None


def test_authenticator_does_not_treat_an_outage_as_an_invalid_token():
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    for handler in (unreachable, lambda request: httpx.Response(500)):
        auth = SupabaseAuthenticator("https://p.supabase.co", "anon", client=_client(handler))
        with pytest.raises(AuthUnavailable):
            auth.verify("token")


def test_supabase_storage_creates_the_bucket_once_then_uploads_and_deletes():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json={})

    storage = SupabaseStorage("https://p.supabase.co", "secret", "docs", client=_client(handler))
    storage.put("alice/1.txt", b"one", "text/plain")
    storage.put("alice/2.txt", b"two", "text/plain")
    storage.delete("alice/1.txt")

    assert [(method, path) for method, path, _ in requests] == [
        ("POST", "/storage/v1/bucket"),
        ("POST", "/storage/v1/object/docs/alice/1.txt"),
        ("POST", "/storage/v1/object/docs/alice/2.txt"),
        ("DELETE", "/storage/v1/object/docs/alice/1.txt"),
    ]
    assert {authorization for _, _, authorization in requests} == {"Bearer secret"}


def test_supabase_storage_reports_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if request.url.path.endswith("/bucket") else 500)

    storage = SupabaseStorage("https://p.supabase.co", "secret", "docs", client=_client(handler))

    with pytest.raises(StorageError):
        storage.put("alice/1.txt", b"one", "text/plain")
    with pytest.raises(StorageError):
        storage.delete("alice/1.txt")


def test_deleting_a_missing_file_is_not_an_error():
    storage = SupabaseStorage(
        "https://p.supabase.co",
        "secret",
        "docs",
        client=_client(lambda request: httpx.Response(404)),
    )

    storage.delete("alice/gone.txt")


def test_local_storage_writes_deletes_and_stays_inside_its_directory(tmp_path):
    storage = LocalStorage(tmp_path / "uploads")

    storage.put("alice/1.txt", b"one", "text/plain")
    assert (tmp_path / "uploads" / "alice" / "1.txt").read_bytes() == b"one"

    storage.delete("alice/1.txt")
    storage.delete("alice/1.txt")
    assert not (tmp_path / "uploads" / "alice" / "1.txt").exists()

    with pytest.raises(StorageError):
        storage.put("../outside.txt", b"x", "text/plain")
    assert not (tmp_path / "outside.txt").exists()


def test_postgres_urls_select_the_psycopg_3_driver():
    engine = make_engine("postgresql://user:pw@db.example.com:5432/postgres")

    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.url.host == "db.example.com"


def test_migrations_build_the_same_schema_as_the_models(tmp_path):
    from sqlalchemy import inspect
    from sqlmodel import SQLModel

    from rag import db

    migrated = make_engine(f"sqlite:///{tmp_path / 'migrated.db'}")
    db.migrate(migrated)
    db.migrate(migrated)  # running it again changes nothing
    direct = make_engine(f"sqlite:///{tmp_path / 'direct.db'}")
    db.create_tables(direct)

    def schema(engine):
        inspector = inspect(engine)
        return {
            table: (
                sorted(
                    (c["name"], str(c["type"]), c["nullable"]) for c in inspector.get_columns(table)
                ),
                sorted(tuple(i["column_names"]) for i in inspector.get_indexes(table)),
                sorted(tuple(u["column_names"]) for u in inspector.get_unique_constraints(table)),
            )
            for table in SQLModel.metadata.tables
        }

    assert schema(migrated) == schema(direct)
