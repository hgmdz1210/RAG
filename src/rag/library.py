"""A user's uploaded documents: add, list and remove, across the three stores.

A document exists in three places: a row in the relational database, the original
file in storage, and its chunks in Qdrant. No transaction spans them, so the row
records the state. It is written first as "processing" and only becomes "ready"
once the chunks are searchable, which makes an interrupted upload visible instead
of leaving chunks nobody can list or delete.
"""

import hashlib
import logging
import mimetypes
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from rag.auth import User
from rag.db import (
    STATUS_FAILED,
    STATUS_READY,
    DocumentRecord,
    MessageRecord,
)
from rag.documents import Document, Hit
from rag.generate import Answer
from rag.loaders import SUPPORTED_SUFFIXES, parse_document
from rag.pipeline import Indexer
from rag.storage import FileStorage, StorageError
from rag.store import QdrantStore

logger = logging.getLogger(__name__)


class UploadRejected(ValueError):
    """The file cannot be accepted; the message is safe to show to the user."""


class DuplicateDocument(Exception):
    """The user has already uploaded a file with the same content."""

    def __init__(self, existing: DocumentRecord) -> None:
        super().__init__(f"Already uploaded as '{existing.filename}'.")
        self.existing = existing


class Library:
    def __init__(
        self,
        engine: Engine,
        storage: FileStorage,
        indexer: Indexer,
        store: QdrantStore,
        *,
        max_upload_bytes: int,
    ) -> None:
        self.engine = engine
        self.storage = storage
        self.indexer = indexer
        self.store = store
        self.max_upload_bytes = max_upload_bytes

    def add(self, user: User, filename: str, data: bytes) -> DocumentRecord:
        """Store, parse and index an uploaded file."""
        # Only the base name is kept: a client can send any path as the file name.
        filename = Path(filename.replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise UploadRejected(f"Unsupported file type. Allowed: {allowed}.")
        if not data:
            raise UploadRejected("The file is empty.")
        if len(data) > self.max_upload_bytes:
            limit_mb = self.max_upload_bytes / 1_000_000
            raise UploadRejected(f"The file is larger than the {limit_mb:.0f} MB limit.")

        try:
            text, title = parse_document(filename, data)
        except Exception as error:
            # Parsers fail in many ways on malformed files.
            logger.warning("Could not parse upload %r", filename, exc_info=True)
            raise UploadRejected("The file could not be read.") from error
        if not text:
            raise UploadRejected("No text could be extracted from the file.")

        record = DocumentRecord(
            owner=user.id,
            filename=filename,
            content_hash=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )
        # The storage key uses ids only, never the file name.
        record.storage_path = f"{user.id}/{record.id}{suffix}"
        with Session(self.engine, expire_on_commit=False) as session:
            session.add(record)
            try:
                session.commit()
            except IntegrityError as error:
                session.rollback()
                existing = session.exec(
                    select(DocumentRecord).where(
                        DocumentRecord.owner == user.id,
                        DocumentRecord.content_hash == record.content_hash,
                    )
                ).one()
                raise DuplicateDocument(existing) from error

        try:
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            self.storage.put(record.storage_path, data, content_type)
            stats = self.indexer.index(
                [Document(doc_id=record.id, text=text, title=title, source=filename, owner=user.id)]
            )
        except Exception as error:
            logger.exception("Indexing upload %s failed", record.id)
            self._cleanup(record)
            return self._update(record.id, status=STATUS_FAILED, error=_describe(error))
        return self._update(record.id, status=STATUS_READY, chunks=stats.chunks)

    def list(self, user: User) -> Sequence[DocumentRecord]:
        with Session(self.engine) as session:
            return session.exec(
                select(DocumentRecord)
                .where(DocumentRecord.owner == user.id)
                .order_by(DocumentRecord.created_at.desc())
            ).all()

    def delete(self, user: User, document_id: str) -> bool:
        """Remove a document everywhere. Returns ``False`` if the user has no such document."""
        with Session(self.engine, expire_on_commit=False) as session:
            record = session.exec(
                select(DocumentRecord).where(
                    DocumentRecord.id == document_id, DocumentRecord.owner == user.id
                )
            ).first()
            if record is None:
                return False
            # Chunks and file go first. If either step fails the row remains, so the
            # document is still listed and the delete can be retried.
            self._cleanup(record, strict=True)
            session.delete(record)
            session.commit()
        return True

    def record_answer(
        self, user: User, question: str, answer: Answer, hits: Sequence[Hit]
    ) -> MessageRecord:
        message = MessageRecord(
            owner=user.id,
            question=question,
            answer=answer.text,
            model=answer.model,
            answerable=answer.answerable,
            sources=[
                {
                    "number": number,
                    "doc_id": hit.chunk.doc_id,
                    "title": hit.chunk.title,
                    "source": hit.chunk.source,
                    "cited": number in answer.cited,
                }
                for number, hit in enumerate(hits, start=1)
            ],
        )
        with Session(self.engine, expire_on_commit=False) as session:
            session.add(message)
            session.commit()
        return message

    def history(self, user: User, limit: int = 50) -> Sequence[MessageRecord]:
        """The user's most recent questions, newest first."""
        with Session(self.engine) as session:
            return session.exec(
                select(MessageRecord)
                .where(MessageRecord.owner == user.id)
                .order_by(MessageRecord.created_at.desc())
                .limit(limit)
            ).all()

    def _cleanup(self, record: DocumentRecord, *, strict: bool = False) -> None:
        try:
            if self.store.count():
                self.store.delete_documents([record.id])
            if record.storage_path:
                self.storage.delete(record.storage_path)
        except Exception:
            if strict:
                raise
            logger.exception("Cleanup of document %s was incomplete", record.id)

    def _update(self, document_id: str, **changes: object) -> DocumentRecord:
        with Session(self.engine, expire_on_commit=False) as session:
            record = session.get(DocumentRecord, document_id)
            for name, value in changes.items():
                setattr(record, name, value)
            session.add(record)
            session.commit()
            return record


def _describe(error: Exception) -> str:
    if isinstance(error, StorageError):
        return str(error)
    return "The document could not be indexed."
