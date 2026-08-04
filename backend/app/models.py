from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    index: int
    section: str
    text: str
    token_count: int
    char_count: int


class DocumentMetadata(BaseModel):
    case_title: str | None = None
    court: str | None = None
    date: str | None = None
    judges: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    source_file: str | None = None


class ChunkStats(BaseModel):
    total_chunks: int
    total_tokens: int
    min_tokens: int
    max_tokens: int
    avg_tokens: float
    sections: list[str]
    tables_extracted: int = 0


class ProcessResult(BaseModel):
    doc_id: str
    metadata: DocumentMetadata
    markdown: str
    chunks: list[Chunk]
    stats: ChunkStats
    cached: bool = False          # served from registry, pipeline not re-run
    pipeline_version: int = 0


class DriveFile(BaseModel):
    id: str
    name: str
    size: int | None = None
    modified_time: str | None = None
