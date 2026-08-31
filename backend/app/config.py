"""Central configuration — named experiment profiles.

A profile bundles every knob that changes retrieval quality: embedding model
(+ its prefixes and dimension), chunking parameters, and index enrichment.
Profiles live in backend/config.yaml so model/chunking combinations can be
declared, indexed side by side, and compared with the eval harness without
touching code.

Learning notes:
- Each profile owns its OWN vector DB (resolve_db_path). Vectors from
  different models live in unrelated spaces — mixing them in one index is
  silent corruption, so isolation is structural, not a convention.
- fingerprint() hashes only the fields that change the VECTORS (model, dim,
  prefixes, chunking, enrichment). Operational knobs (base_url, batch_size,
  timeout) are excluded — changing them must not invalidate an index.
- extra="forbid" on every model: a typo'd key in config.yaml fails loudly at
  startup instead of silently falling back to a default.
- Selection order: explicit argument > VAKIL_PROFILE env > active_profile in
  the file. CLIs take --profile; the API server uses the active profile.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BACKEND_DIR / "config.yaml"


def _load_dotenv(path: Path = BACKEND_DIR / ".env") -> None:
    """Load backend/.env into os.environ at import, so CLIs (`python -m
    app.llm`, indexer, evals) see the same vars as `make backend` (whose
    shell sources the file). setdefault = real environment always wins —
    `VAKIL_CHAT_MODEL=x python -m ...` still overrides the file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


class ConfigError(RuntimeError):
    pass


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Registry name of an Embedder implementation (embeddings._EMBEDDERS).
    # A plain string, not a Literal: adding a provider must not require
    # touching this schema — the factory validates against the registry.
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    dim: int = Field(768, ge=1)
    batch_size: int = Field(64, ge=1)
    timeout_s: float = Field(120.0, gt=0)
    # Prefix-trained models (nomic, mxbai) need distinct document/query
    # prefixes; leave empty for symmetric models.
    document_prefix: str = ""
    query_prefix: str = ""


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Registry name of a Chunker implementation (chunker._CHUNKERS).
    strategy: str = "section-aware"
    target_tokens: int = Field(700, ge=1)
    max_tokens: int = Field(900, ge=1)
    min_tokens: int = Field(120, ge=0)
    overlap_tokens: int = Field(80, ge=0)
    encoding: str = "cl100k_base"

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ChunkingConfig":
        if not self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("require overlap_tokens < target_tokens <= max_tokens")
        if self.min_tokens >= self.target_tokens:
            raise ValueError("require min_tokens < target_tokens")
        return self


class IndexingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Prepend "case title — section" to the EMBEDDED text (raw text is stored
    # for display either way). A comparison knob: measure its lift in evals.
    enrich: bool = True


class RerankConfig(BaseModel):
    """Cross-encoder precision stage over the fused candidate pool.
    Orthogonal to strategy (a flag, not a Literal value): a precision stage
    doesn't care whether the pool came from dense or hybrid retrieval."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Registry name of a Reranker implementation (rerank._RERANKERS).
    provider: str = "cross-encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Fused candidates fed to the cross-encoder (the "shortlist" size).
    pool: int = Field(50, ge=1)
    batch_size: int = Field(16, ge=1)


class RewriteConfig(BaseModel):
    """Query rewriting knobs (hybrid-only — rewrites exist to feed lexical
    lists). Separate booleans deliberately: the eval matrix turns each cure
    on independently to attribute improvements to the right mechanism.
    Defaults OFF so profiles that never mention these keys retrieve
    identically to before 2c."""

    model_config = ConfigDict(extra="forbid")

    # Regex citation detector -> FTS5 phrase queries (the q017 cure).
    citations: bool = False
    # LLM keyword rewrite via app/llm (the q004 cure).
    llm: bool = False
    # generation.models key for the rewrite; "" = active/env chain.
    model: str = ""


class RetrievalConfig(BaseModel):
    """Query-time retrieval strategy. Deliberately OUTSIDE fingerprint():
    the lexical (FTS5) index is derived from payloads the store already
    holds, so flipping dense<->hybrid never invalidates vectors — it only
    changes how the same index is queried."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["dense", "hybrid"] = "dense"
    # Depth each ranker contributes BEFORE fusion (retrieve wide, fuse, cut
    # to k). Only used by hybrid.
    candidates: int = Field(50, ge=1)
    # RRF constant: flattens the top of 1/(k+rank) so single-list rank noise
    # can't dominate; 60 is the empirical default from Cormack et al. 2009.
    rrf_k: int = Field(60, ge=0)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""  # filled from the profiles-map key
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    # Registry name of a VectorIndex implementation (vector_store._STORES).
    store: str = "sqlite"
    db_path: str | None = None  # relative to backend/; default vectors-<name>.db

    def resolve_db_path(self) -> Path:
        rel = self.db_path or f"vectors-{self.name}.db"
        return (BACKEND_DIR / rel).resolve()

    def fingerprint(self) -> str:
        """Identity of the vectors this profile produces (see module notes)."""
        identity = {
            "model": self.embedding.model,
            "dim": self.embedding.dim,
            "document_prefix": self.embedding.document_prefix,
            "query_prefix": self.embedding.query_prefix,
            "chunking": self.chunking.model_dump(),
            "enrich": self.indexing.enrich,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return digest[:12]

    def snapshot(self) -> dict:
        """Full config dump for embedding into eval results (reproducibility)."""
        return self.model_dump() | {"fingerprint": self.fingerprint()}


class GenerationConfig(BaseModel):
    """One named chat model. Generation is TOP-LEVEL, not per-profile: the
    chat model reads retrieved text, it never produces vectors, so swapping
    it can't invalidate an index and it stays out of profile fingerprints."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""  # filled from the generation.models map key
    # Registry name of a ChatModel implementation (llm._CHAT_MODELS).
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    # Sized for RAG prompts: system + k chunks + question + answer headroom.
    # Ollama TRUNCATES the prompt front silently on overflow — keep generous.
    num_ctx: int = Field(8192, ge=1024)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(300.0, gt=0)
    # For API providers (openai-compatible): NAME of the env var holding the
    # key — never the key itself; config.yaml is committed.
    api_key_env: str = ""


class GenerationSection(BaseModel):
    """Named chat models + the active switch — same shape as profiles:
    declare several, flip `active` (or VAKIL_CHAT_MODEL env) to switch."""

    model_config = ConfigDict(extra="forbid")

    active: str
    models: dict[str, GenerationConfig]

    @model_validator(mode="after")
    def _wire_names(self) -> "GenerationSection":
        for key, model in self.models.items():
            model.name = key
        if self.active not in self.models:
            raise ValueError(
                f"generation.active '{self.active}' not in models "
                f"{sorted(self.models)}"
            )
        return self


class AgentSection(BaseModel):
    """Which agent loop answers /agent/ask — top-level like generation: the
    loop only orchestrates tools, it never produces vectors, so switching
    can't invalidate an index. Registry names in agent/__init__.py."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "langgraph"  # hand-rolled | langgraph


class ComponentsConfig(BaseModel):
    """Pipeline-LEVEL component selection — parser/metadata output in output/
    is shared by every profile, so these are global, not per-profile.
    Values are registry names in the implementing modules."""

    model_config = ConfigDict(extra="forbid")

    parser: str = "docling"
    metadata_extractor: str = "regex"
    # Citation-graph extractor (3c). NOT under PIPELINE_VERSION: edges are
    # re-derived from persisted markdown via `python -m app.citations.backfill`.
    citation_extractor: str = "regex"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_profile: str
    agent: AgentSection = Field(default_factory=AgentSection)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    generation: GenerationSection = Field(
        default_factory=lambda: GenerationSection(
            active="llama", models={"llama": GenerationConfig()}
        )
    )
    profiles: dict[str, Profile]

    @model_validator(mode="after")
    def _wire_names(self) -> "AppConfig":
        for key, profile in self.profiles.items():
            profile.name = key
        if self.active_profile not in self.profiles:
            raise ValueError(
                f"active_profile '{self.active_profile}' not in profiles "
                f"{sorted(self.profiles)}"
            )
        return self


def _config_path() -> Path:
    return Path(os.environ.get("VAKIL_CONFIG", DEFAULT_CONFIG_PATH))


@lru_cache(maxsize=4)
def _load(path_str: str) -> AppConfig:
    path = Path(path_str)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return AppConfig(**data)
    except (yaml.YAMLError, ValueError) as e:
        raise ConfigError(f"invalid config {path}: {e}") from e


def get_config() -> AppConfig:
    return _load(str(_config_path()))


def get_chat_config(name: str | None = None) -> GenerationConfig:
    """Resolve a chat model: explicit name > VAKIL_CHAT_MODEL env > active."""
    gen = get_config().generation
    resolved = name or os.environ.get("VAKIL_CHAT_MODEL") or gen.active
    if resolved not in gen.models:
        raise ConfigError(
            f"unknown chat model '{resolved}'; available: {sorted(gen.models)}"
        )
    return gen.models[resolved]


def get_agent_kind(name: str | None = None) -> str:
    """Resolve the agent loop: explicit name > VAKIL_AGENT env > config."""
    return name or os.environ.get("VAKIL_AGENT") or get_config().agent.kind


def get_profile(name: str | None = None) -> Profile:
    """Resolve a profile: explicit name > VAKIL_PROFILE env > active_profile."""
    cfg = get_config()
    resolved = name or os.environ.get("VAKIL_PROFILE") or cfg.active_profile
    if resolved not in cfg.profiles:
        raise ConfigError(
            f"unknown profile '{resolved}'; available: {sorted(cfg.profiles)}"
        )
    return cfg.profiles[resolved]
