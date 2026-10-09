"""Alembic environment. The engine is supplied by ``rag.db.migrate``."""

from alembic import context
from sqlmodel import SQLModel

from rag import db  # noqa: F401  (registers the tables on the metadata)

engine = context.config.attributes["engine"]

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=SQLModel.metadata)
    with context.begin_transaction():
        context.run_migrations()
    connection.commit()
