"""Runtime settings, read once from environment variables (prefix ``BIORETRIEVAL_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"BIORETRIEVAL_{name}", default)


def _path(name: str) -> Path | None:
    value = _env(name)
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class Settings:
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", "~/.cache/bioretrieval-mcp")).expanduser())
    # HTTP behaviour
    http_timeout_s: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT_S", "20")))
    http_max_retries: int = field(default_factory=lambda: int(_env("HTTP_MAX_RETRIES", "3")))
    http_backoff_s: float = field(default_factory=lambda: float(_env("HTTP_BACKOFF_S", "0.5")))
    http_cache_ttl_s: int = field(default_factory=lambda: int(_env("HTTP_CACHE_TTL_S", str(24 * 3600))))
    # NCBI asks for tool/email identification and allows 10 req/s with an API key (3 without).
    ncbi_api_key: str | None = field(default_factory=lambda: os.environ.get("NCBI_API_KEY"))
    ncbi_email: str | None = field(default_factory=lambda: _env("NCBI_EMAIL"))
    # ML artifacts
    esm_model: str = field(default_factory=lambda: _env("ESM_MODEL", "esm2_t33_650M_UR50D"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    retrieval_checkpoint: Path | None = field(default_factory=lambda: _path("RETRIEVAL_CHECKPOINT"))
    map_index: Path | None = field(default_factory=lambda: _path("MAP_INDEX"))
    cpgpt_embeddings: Path | None = field(default_factory=lambda: _path("CPGPT_EMBEDDINGS"))

    @property
    def http_cache_path(self) -> Path:
        return self.cache_dir / "http_cache.sqlite"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.cache_dir / "esm"


def get_settings() -> Settings:
    return Settings()
