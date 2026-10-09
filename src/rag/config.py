"""Runtime settings, read from ``RAG_*`` environment variables or a local ``.env`` file."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.fusion import RRF_K

SearchMode = Literal["dense", "sparse", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Storage. With no URL the client runs Qdrant embedded, persisting to `qdrant_path`.
    qdrant_url: str | None = None
    qdrant_path: str = "data/qdrant"
    collection: str = "documents"

    # Models, all served through ONNX Runtime by fastembed.
    dense_model: str = "BAAI/bge-small-en-v1.5"
    # BGE v1.5 is trained with this instruction on the query side only.
    dense_query_prefix: str = "Represent this sentence for searching relevant passages: "
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str = "data/models"

    # Chunking, in characters.
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # Retrieval defaults; each can be overridden per request.
    search_mode: SearchMode = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=30, ge=1, le=200)
    # Constant in reciprocal rank fusion. Smaller values let top ranks dominate.
    rrf_k: int = Field(default=RRF_K, ge=0)
    # Off by default: on the benchmark in results/ the cross-encoder cost about two
    # seconds per query on CPU without improving the ranking. Measure before enabling.
    rerank: bool = False

    # Answer generation.
    llm_model: str = "gemini-3.8-flash"
    llm_max_tokens: int = Field(default=8192, gt=0)
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "RAG_GEMINI_API_KEY"),
    )

    # Relational database for the document registry and the question history.
    # SQLite by default; a postgresql:// URL (for example Supabase) for real use.
    database_url: str = "sqlite:///data/app.db"

    # Supabase project. With the URL and the anon key set, the API requires a
    # signed-in user and keeps each user's documents apart. With the service key
    # as well, uploaded files are kept in Supabase Storage instead of on disk.
    supabase_url: str | None = Field(default=None, validation_alias="SUPABASE_URL")
    supabase_anon_key: SecretStr | None = Field(default=None, validation_alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: SecretStr | None = Field(
        default=None, validation_alias="SUPABASE_SERVICE_ROLE_KEY"
    )
    storage_bucket: str = "documents"
    upload_dir: str = "data/uploads"
    max_upload_mb: int = Field(default=20, gt=0, le=200)

    # When set, every API route except /health requires this value in X-API-Key.
    # Ignored when Supabase sign-in is configured.
    api_key: SecretStr | None = None

    @field_validator(
        "qdrant_url",
        "gemini_api_key",
        "api_key",
        "supabase_url",
        "supabase_anon_key",
        "supabase_service_role_key",
        mode="before",
    )
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        # `KEY=` in a .env file arrives as an empty string, not as a missing value.
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
