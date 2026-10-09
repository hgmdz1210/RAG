"""Plain data types passed between the pipeline stages."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    # Id of the user the document belongs to; empty for a shared corpus.
    owner: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    title: str = ""
    source: str = ""
    owner: str = ""

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedders: the chunk prefixed with its document title.

        A chunk cut from the middle of a document often no longer names its subject;
        the title puts it back.
        """
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
