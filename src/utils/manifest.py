"""Append-only provenance log for pipeline runs (protocol Step 12)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def iter_manifest_records(manifest_path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def latest_manifest_record(
    manifest_path: Path | str,
    backbone: str,
    *,
    step: str | None = None,
) -> dict[str, Any] | None:
    """Return the last manifest line for ``backbone``, optionally filtered by ``step``."""
    latest: dict[str, Any] | None = None
    for record in iter_manifest_records(manifest_path):
        if record.get("backbone") != backbone:
            continue
        if step is not None and record.get("step") != step:
            continue
        latest = record
    return latest


def append_manifest(record: dict[str, Any], manifest_path: Path | str) -> None:
    """Append one JSON object as a line to results/manifest.jsonl."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in record:
        record = {
            **record,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
