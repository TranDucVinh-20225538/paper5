"""Append-only provenance log for pipeline runs (protocol Step 12)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
