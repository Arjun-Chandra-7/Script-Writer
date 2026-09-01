from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def build_shards(examples: Iterable[dict[str, Any]], output_dir: Path, *, shard_size: int = 1000) -> dict[str, Any]:
    if shard_size <= 0: raise ValueError("shard_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard, count, entries = [], 0, []
    def flush(items: list[dict[str, Any]]) -> None:
        nonlocal count
        payload = ("\n".join(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for x in items) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        path = output_dir / f"shard-{len(entries):05d}-{digest[:16]}.jsonl"
        if not path.exists():
            fd, temporary = tempfile.mkstemp(prefix=".shard-", suffix=".part", dir=output_dir)
            with os.fdopen(fd, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)
        entries.append({"path": path.name, "sha256": digest, "count": len(items)})
        count += len(items)
    for example in examples:
        shard.append(example)
        if len(shard) == shard_size: flush(shard); shard=[]
    if shard: flush(shard)
    manifest = {"format":"viralyst-sharded-training-v1", "shard_size":shard_size, "example_count":count, "shards":entries}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    path = output_dir / f"manifest-{manifest['manifest_sha256'][:16]}.json"
    if not path.exists(): path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return {**manifest,"manifest_path":str(path)}
