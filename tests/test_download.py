import hashlib

import httpx
import pytest

from production_rag.core.eval.download import fetch


def _client(seen: list, status: int, body: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_caches_and_checks_sha(tmp_path):
    body = b"hello"
    seen: list = []
    client = _client(seen, 200, body)
    p = fetch(
        "https://x/a.txt",
        sha256=hashlib.sha256(body).hexdigest(),
        cache=tmp_path,
        client=client,
    )
    assert p.read_bytes() == body
    assert p.name.endswith("-a.txt")
    assert seen[0].headers["user-agent"] == "production-rag-eval"
    fetch("https://x/a.txt", cache=tmp_path, client=client)
    assert len(seen) == 1


def test_fetch_checksum_mismatch_removes_file(tmp_path):
    with pytest.raises(ValueError, match="checksum mismatch for https://x/b.txt"):
        fetch("https://x/b.txt", sha256="0" * 64, cache=tmp_path, client=_client([], 200, b"x"))
    assert list(tmp_path.iterdir()) == []


def test_fetch_http_error(tmp_path):
    with pytest.raises(RuntimeError, match="download failed 404 https://x/c.txt"):
        fetch("https://x/c.txt", cache=tmp_path, client=_client([], 404, b""))
    assert list(tmp_path.iterdir()) == []


def test_fetch_rechecks_sha_of_cached_file(tmp_path):
    body = b"hello"
    good = hashlib.sha256(body).hexdigest()
    client = _client([], 200, body)
    path = fetch("https://x/d.txt", sha256=good, cache=tmp_path, client=client)
    assert fetch("https://x/d.txt", sha256=good, cache=tmp_path, client=client) == path
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch("https://x/d.txt", sha256=good, cache=tmp_path, client=client)
