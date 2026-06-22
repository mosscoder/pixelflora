"""Global configuration: non-secret settings from ``config.toml`` (TOML, per the
project's convention), secrets from environment variables only."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _find_config(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    if os.environ.get("PIXELFLORA_CONFIG"):
        return Path(os.environ["PIXELFLORA_CONFIG"])
    # walk up from cwd looking for config.toml
    here = Path.cwd()
    for d in [here, *here.parents]:
        cand = d / "config.toml"
        if cand.exists():
            return cand
    return None


@dataclass
class Config:
    user_agent: str = "pixelflora/0.1"
    contact_email: str = ""
    timeout_s: int = 60
    max_retries: int = 4
    rate_limit_s: float = 1.0          # iNaturalist best practice: <= 1 req/s
    download_workers: int = 10         # photo files come from a bulk CDN/S3, fetch them concurrently
    download_rate_limit_s: float = 0.1  # gentle per-host pacing for photo fetches so bulk runs don't trip the static.inaturalist.org CDN
    cache_dir: str = ".pixelflora_cache"
    publish_private: bool = True
    publish_owner: str = ""
    publish_push: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    # secrets (env only)
    @property
    def hf_token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        p = _find_config(path)
        if p and p.exists():
            data = tomllib.loads(p.read_text())
            cfg.raw = data
            http = data.get("http", {})
            cfg.user_agent = http.get("user_agent", cfg.user_agent)
            cfg.contact_email = http.get("contact_email", cfg.contact_email)
            cfg.timeout_s = http.get("timeout_s", cfg.timeout_s)
            cfg.max_retries = http.get("max_retries", cfg.max_retries)
            cfg.rate_limit_s = http.get("rate_limit_s", cfg.rate_limit_s)
            cfg.download_workers = http.get("download_workers", cfg.download_workers)
            cfg.download_rate_limit_s = http.get("download_rate_limit_s", cfg.download_rate_limit_s)
            cfg.cache_dir = data.get("cache", {}).get("dir", cfg.cache_dir)
            pub = data.get("publish", {})
            cfg.publish_private = pub.get("private", cfg.publish_private)
            cfg.publish_owner = pub.get("owner", cfg.publish_owner)
            cfg.publish_push = pub.get("push", cfg.publish_push)
        return cfg
