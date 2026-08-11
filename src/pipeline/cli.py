"""CLI entrypoint for scripts/run_all.sh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper 5 per-backbone pipeline")
    parser.add_argument("config", type=Path, help="configs/<backbone>.yaml")
    parser.add_argument(
        "--through-step",
        type=int,
        default=3,
        choices=range(0, 13),
        help="Run through this protocol step (default: 3)",
    )
    parser.add_argument(
        "--loader",
        default=None,
        help="Override backbone.loader (e.g. fixture for tests)",
    )
    parser.add_argument(
        "--skip-split-check",
        action="store_true",
        help="Skip split checksum gate (tests only)",
    )
    parser.add_argument(
        "--allow-null-preprocessing-hash",
        action="store_true",
        help="Allow null preprocessing.sha256 (tests only)",
    )
    parser.add_argument(
        "--allow-null-checkpoint",
        action="store_true",
        help="Allow null checkpoint (tests only)",
    )
    args = parser.parse_args(argv)

    if args.through_step < 3:
        from src.pipeline.hard_stops import run_step0_hard_stops
        from src.utils.config import load_backbone_config

        cfg = load_backbone_config(args.config)
        repo_root = args.config.resolve().parents[1]
        run_step0_hard_stops(
            cfg,
            repo_root,
            require_preprocessing_hash=not args.allow_null_preprocessing_hash,
            require_split_checksum=not args.skip_split_check,
            require_checkpoint=not args.allow_null_checkpoint,
        )
        print(f"Step 0 passed for {cfg.name}")
        return 0

    if args.through_step == 3:
        from src.pipeline.steps import run_steps_0_through_3

        record = run_steps_0_through_3(
            args.config,
            require_preprocessing_hash=not args.allow_null_preprocessing_hash,
            require_split_checksum=not args.skip_split_check,
            require_checkpoint=not args.allow_null_checkpoint,
            loader_override=args.loader,
            fixture_train_n=8,
            fixture_eval_n=4,
        )
        print(f"Step 3 complete: {record['backbone']} train_sha256={record['train_sha256'][:12]}…")
        return 0

    print(f"Steps 4–{args.through_step} not implemented yet.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
