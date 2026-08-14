#!/usr/bin/env python3
"""Export manifest.jsonl lines for one backbone (for cross-machine GPU→CPU handoff)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.manifest import iter_manifest_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export manifest lines for one backbone")
    parser.add_argument("backbone", help="backbone name, e.g. medsam")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results" / "manifest.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args(argv)

    rows = [r for r in iter_manifest_records(args.manifest) if r.get("backbone") == args.backbone]
    if not rows:
        print(f"No manifest lines for backbone {args.backbone!r}", file=sys.stderr)
        return 1

    text = "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(rows)} lines → {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
