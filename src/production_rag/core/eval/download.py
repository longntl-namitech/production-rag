"""Download a public file once and reuse it from a local cache."""

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

CACHE = Path.home() / ".cache" / "production-rag"


def fetch(
    url: str,
    *,
    sha256: str | None = None,
    cache: Path | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Return the cached file for `url`, downloading it first when it is not there."""
    cache = cache or CACHE
    cache.mkdir(parents=True, exist_ok=True)
    basename = Path(urlparse(url).path).name or "download"
    path = cache / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}-{basename}"
    if path.exists():
        return path
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=120)
    fd, tmp_name = tempfile.mkstemp(dir=cache, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            with client.stream(
                "GET", url, headers={"User-Agent": "production-rag-eval"}, follow_redirects=True
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"download failed {resp.status_code} {url}")
                for block in resp.iter_bytes():
                    out.write(block)
        if sha256 and hashlib.sha256(tmp.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"checksum mismatch for {url}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
        if own:
            client.close()
    return path
