"""A small on-disk cache for LLM text that is costly to make again.

One JSON file per entry, `{"value": "..."}`, under `root/<first two key chars>/<key>.json`.
The key is a hash of the model, the prompt and the input text, so changing any of them is a
miss. The default folder is git-ignored.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path


class DiskCache:
    def __init__(self, root: Path = Path(".cache/llm")):
        self.root = Path(root)

    @staticmethod
    def key(model: str, prompt: str, text: str) -> str:
        return hashlib.sha256(json.dumps([model, prompt, text]).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        try:
            value = json.loads(self._path(key).read_text(encoding="utf-8"))["value"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return value if isinstance(value, str) else None

    def put(self, key: str, value: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{key}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps({"value": value}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
