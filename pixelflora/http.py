"""A small polite HTTP client: shared User-Agent, per-host rate limiting,
retries with exponential backoff, and a tiny on-disk JSON cache for API GETs
(used for reproducible replay of harvests)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import requests


class PoliteClient:
    def __init__(self, *, user_agent: str, timeout_s: int = 60, max_retries: int = 4,
                 rate_limit_s: float = 0.34, cache_dir: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.rate_limit_s = rate_limit_s
        self.cache_dir = Path(cache_dir) / "http" if cache_dir else None
        self._last_hit: dict[str, float] = {}

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        now = time.monotonic()
        wait = self.rate_limit_s - (now - self._last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def _cache_path(self, url: str, params: Optional[dict]) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def get_json(self, url: str, params: Optional[dict] = None, *, use_cache: bool = True) -> Any:
        cp = self._cache_path(url, params)
        if use_cache and cp and cp.exists():
            return json.loads(cp.read_text())
        data = self._request("GET", url, params=params).json()
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(data))
        return data

    def _request(self, method: str, url: str, **kw) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._throttle(url)
            try:
                resp = self.session.request(method, url, timeout=self.timeout_s, **kw)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{resp.status_code} from {url}")
                resp.raise_for_status()
                return resp
            except (requests.RequestException, ) as e:
                last_exc = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"request failed after {self.max_retries} attempts: {url}") from last_exc

    def get_bytes(self, url: str) -> bytes:
        return self._request("GET", url, stream=False).content

    def post_json(self, url: str, payload: dict, *, auth: Optional[tuple] = None) -> requests.Response:
        return self._request("POST", url, json=payload, auth=auth,
                             headers={"Content-Type": "application/json"})
