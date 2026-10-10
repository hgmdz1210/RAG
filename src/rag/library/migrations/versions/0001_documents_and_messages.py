"""Create the documents and messages tables.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TABLES = ("documents", "messages")


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("owner", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("chunks", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("storage_path", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner", "content_hash", name="uq_documents_owner_hash"),
    )
    op.create_index("ix_documents_owner", "documents", ["owner"])

    op.create_table(
        "messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("owner", sa.String(), nullable=False),
        sa.Column("question", sa.String(), nullable=False),
        sa.Column("answer", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_messages_owner", "messages", ["owner"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])

    if op.get_bind().dialect.name == "postgresql":
        # Supabase exposes every table in the public schema through its REST API.
        # Row level security with no policies denies that route entirely; the
        # backend connects as the table owner, which the policies do not restrict.
        # alembic_version is Alembic's own bookkeeping table, created before this runs.
        for table in (*TABLES, "alembic_version"):
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("documents")
