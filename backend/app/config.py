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
from pydantic import BaseModel, ConfigDict, Field, model_validator

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BACKEND_DIR / "config.yaml"


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


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""  # filled from the profiles-map key
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
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


class ComponentsConfig(BaseModel):
    """Pipeline-LEVEL component selection — parser/metadata output in output/
    is shared by every profile, so these are global, not per-profile.
    Values are registry names in the implementing modules."""

    model_config = ConfigDict(extra="forbid")

    parser: str = "docling"
    metadata_extractor: str = "regex"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_profile: str
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
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


def get_profile(name: str | None = None) -> Profile:
    """Resolve a profile: explicit name > VAKIL_PROFILE env > active_profile."""
    cfg = get_config()
    resolved = name or os.environ.get("VAKIL_PROFILE") or cfg.active_profile
    if resolved not in cfg.profiles:
        raise ConfigError(
            f"unknown profile '{resolved}'; available: {sorted(cfg.profiles)}"
        )
    return cfg.profiles[resolved]
