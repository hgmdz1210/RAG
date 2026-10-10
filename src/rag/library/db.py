"""Relational data: the registry of uploaded documents and the question history.

Vectors and chunk text live in Qdrant. These tables hold what has to be listed,
counted and deleted per user.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column, DateTime, UniqueConstraint
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, SQLModel, create_engine

STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class DocumentRecord(SQLModel, table=True):
    """One uploaded file. Its id is also the ``doc_id`` of its chunks in Qdrant."""

    __tablename__ = "documents"
    # The same user uploading the same bytes twice is a duplicate, not a new document.
    __table_args__ = (UniqueConstraint("owner", "content_hash", name="uq_documents_owner_hash"),)

    id: str = Field(default_factory=_new_id, primary_key=True)
    owner: str = Field(index=True)
    filename: str
    content_hash: str
    size_bytes: int
    chunks: int = 0
    # "processing" until the chunks are in Qdrant, so an interrupted upload is visible.
    status: str = STATUS_PROCESSING
    error: str | None = None
    storage_path: str | None = None
    created_at: datetime = Field(
        default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class MessageRecord(SQLModel, table=True):
    """One question and the answer it received."""

    __tablename__ = "messages"

    id: str = Field(default_factory=_new_id, primary_key=True)
    owner: str = Field(index=True)
    question: str
    answer: str
    model: str = ""
    answerable: bool = True
    # The passages shown with the answer: number, doc_id, title, source, cited.
    sources: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


def make_engine(url: str) -> Engine:
    """Create an engine for PostgreSQL or SQLite."""
    if url.startswith("postgresql://"):
        # Select the psycopg 3 driver; a bare postgresql:// URL means psycopg2.
        url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
    if url.startswith("sqlite"):
        options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url or url == "sqlite://":
            # One shared connection, otherwise every session sees an empty database.
            options["poolclass"] = StaticPool
        return create_engine(url, **options)
    # pre_ping replaces connections that a pooler in front of the database dropped.
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def create_tables(engine: Engine) -> None:
    """Create missing tables directly. Used for SQLite; PostgreSQL uses migrations."""
    SQLModel.metadata.create_all(engine)


def migrate(engine: Engine) -> None:
    """Bring a database to the current schema by running the bundled migrations."""
    from importlib.resources import files

    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(files("rag").joinpath("library/migrations")))
    config.attributes["engine"] = engine
    command.upgrade(config, "head")
