"""CLI entrypoint for scripts/run_all.sh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper 5 per-backbone pipeline")
    parser.add_argument("config", type=Path, help="configs/<backbone>.yaml")
    parser.add_argument(
        "--from-step",
        type=int,
        default=0,
        choices=range(0, 13),
        help="Start at this protocol step (default: 0). Use 8 to resume CPU analysis after GPU phase.",
    )
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
    parser.add_argument(
        "--accept-legacy-selection",
        action="store_true",
        help="Allow --from-step 7 to use a step-6 record that predates grid_epochs logging.",
    )
    parser.add_argument(
        "--grid-epochs",
        type=int,
        default=None,
        help="Override intervention epochs during Step 5 grid (tests only)",
    )
    args = parser.parse_args(argv)

    common = dict(
        require_preprocessing_hash=not args.allow_null_preprocessing_hash,
        require_split_checksum=not args.skip_split_check,
        require_checkpoint=not args.allow_null_checkpoint,
        loader_override=args.loader,
    )

    if args.from_step > args.through_step:
        parser.error("--from-step must be <= --through-step")

    # Resume at steps 4-6 without re-extracting. Every piece already existed
    # (resolve_extraction, run_steps_4_through_6); only this wiring was missing, so
    # --from-step was accepted, validated, and then silently ignored for 1-7 — the
    # dispatch below keys on --through-step alone and always starts at 0. For MedSAM
    # that meant ~11 hours of re-extraction on every attempt to run step 5.
    if 4 <= args.from_step <= 6:
        if args.through_step > 6:
            parser.error("--from-step 4..6 currently runs through step 6 only")
        from src.pipeline.steps import resolve_extraction, run_steps_4_through_6
        from src.utils.config import load_backbone_config

        # find_repo_root is the module-level wrapper below — importing it here as well
        # would make the name function-local for the whole of main(), leaving it unbound
        # in every branch that does not run this one.
        cfg = load_backbone_config(args.config)
        root = find_repo_root(args.config)
        extraction = resolve_extraction(root, cfg)
        print(
            f"Resuming at step {args.from_step} from existing embeddings: "
            f"{extraction.train_n}x{extraction.embed_dim} train, {extraction.eval_n} eval"
        )
        record = run_steps_4_through_6(cfg, extraction, root, grid_epochs=args.grid_epochs)
        print(
            f"Step 6 complete: {record['backbone']} r={record['selected_r']} "
            f"lambda={record['selected_lambda_proj']} gate0=PASS"
        )
        return 0

    # Resume at step 7 (train arms -> CPU phase) using the (r, lambda_proj) that step 5
    # selected, read back from the manifest.
    if args.from_step == 7:
        from src.pipeline.steps import resolve_extraction, run_steps_7_through_12
        from src.utils.config import load_backbone_config
        from src.utils.manifest import latest_manifest_record

        cfg = load_backbone_config(args.config)
        root = find_repo_root(args.config)
        rec = latest_manifest_record(root / "results" / "manifest.jsonl", cfg.name, step="6_gate0")
        if not rec or rec.get("selected_r") is None:
            parser.error(
                f"{cfg.name}: no step-6 record with a selected (r, lambda_proj). "
                "Run --from-step 4 --through-step 6 first."
            )

        # Protocol Step 5 selects from the full pre-committed grid. Inheriting a pair
        # chosen by a shortened grid would silently substitute a different selection
        # rule, so refuse rather than warn.
        # Refuse when the selection cannot be confirmed, rather than warn. A warning
        # here would be useless in exactly the case that matters: the first record this
        # guard met was a 1-epoch smoke run, and because legacy records carry no
        # grid_epochs at all it would have sailed through on a printed caution.
        MISSING = object()
        grid_epochs = rec.get("grid_epochs", MISSING)
        if isinstance(grid_epochs, int):
            parser.error(
                f"{cfg.name}: the step-6 record was produced with --grid-epochs "
                f"{grid_epochs}, a shortened grid. Its r={rec['selected_r']}, "
                f"lambda_proj={rec['selected_lambda_proj']} are not a protocol selection. "
                "Re-run --from-step 4 --through-step 6 without --grid-epochs."
            )
        if grid_epochs is MISSING and not args.accept_legacy_selection:
            parser.error(
                f"{cfg.name}: the step-6 record predates grid_epochs logging, so its "
                f"r={rec['selected_r']}, lambda_proj={rec['selected_lambda_proj']} cannot be "
                "confirmed as a full-grid protocol selection. Either re-run "
                "--from-step 4 --through-step 6, or pass --accept-legacy-selection to "
                "assert the record is sound."
            )

        extraction = resolve_extraction(root, cfg)
        print(
            f"Resuming at step 7: r={rec['selected_r']} "
            f"lambda_proj={rec['selected_lambda_proj']} from step-6 record"
        )
        record = run_steps_7_through_12(
            cfg, extraction, root,
            r=int(rec["selected_r"]),
            lambda_proj=float(rec["selected_lambda_proj"]),
            train_epochs=args.grid_epochs,
        )
        if record.get("gate1") == "not_testable":
            print(f"Step 9: {record['backbone']} Gate 1 NOT TESTABLE — stopped.", file=sys.stderr)
            return 2
        print(f"Step 12 complete: {record['backbone']}")
        return 0

    if args.from_step in (1, 2, 3):
        # Fail loudly rather than quietly restarting from 0, which is what happened
        # before and is indistinguishable from success until the wall-clock bill arrives.
        parser.error(
            f"--from-step {args.from_step} is not supported. "
            "Use 0 (full run), 4..6 (grid search), 7 (train arms onward), or 8 (CPU phase)."
        )

    if args.from_step >= 8:
        if args.through_step < 8:
            parser.error("CPU phase requires --through-step >= 8")
        from src.pipeline.steps import run_steps_8_through_12_from_config

        record = run_steps_8_through_12_from_config(args.config)
        if record.get("gate1") == "not_testable":
            print(f"Step 9: {record['backbone']} Gate 1 NOT TESTABLE — stopped.", file=sys.stderr)
            return 2
        print(f"Step {args.through_step} complete: {record['backbone']} gate1=PASS")
        return 0

    if args.through_step < 3:
        from src.pipeline.hard_stops import run_step0_hard_stops
        from src.utils.config import load_backbone_config

        cfg = load_backbone_config(args.config)
        repo_root = find_repo_root(args.config)
        run_step0_hard_stops(cfg, repo_root, **{k: v for k, v in common.items() if k != "loader_override"})
        print(f"Step 0 passed for {cfg.name}")
        return 0

    if args.through_step == 3:
        from src.pipeline.steps import run_steps_0_through_3

        record = run_steps_0_through_3(
            args.config,
            fixture_train_n=8,
            fixture_eval_n=4,
            **common,
        )
        print(f"Step 3 complete: {record['backbone']} train_sha256={record['train_sha256'][:12]}…")
        return 0

    if args.through_step <= 6:
        from src.pipeline.steps import run_steps_0_through_6

        record = run_steps_0_through_6(
            args.config,
            fixture_train_n=200,
            fixture_eval_n=120,
            grid_epochs=args.grid_epochs,
            **common,
        )
        print(
            f"Step 6 complete: {record['backbone']} r={record['selected_r']} "
            f"lambda={record['selected_lambda_proj']} gate0=PASS"
        )
        return 0

    if args.through_step <= 7:
        from src.pipeline.steps import run_steps_0_through_7

        record = run_steps_0_through_7(
            args.config,
            fixture_train_n=200,
            fixture_eval_n=120,
            grid_epochs=args.grid_epochs,
            train_epochs=args.grid_epochs,
            **common,
        )
        print(
            f"Step 7 complete: {record['backbone']} r={record['r']} "
            f"lambda={record['lambda_proj']} (GPU phase done)"
        )
        return 0

    if args.through_step <= 12:
        from src.pipeline.steps import run_steps_0_through_12

        record = run_steps_0_through_12(
            args.config,
            fixture_train_n=200,
            fixture_eval_n=120,
            grid_epochs=args.grid_epochs,
            train_epochs=args.grid_epochs,
            **common,
        )
        if record.get("gate1") == "not_testable":
            print(f"Step 9: {record['backbone']} Gate 1 NOT TESTABLE — stopped.", file=sys.stderr)
            return 2
        print(f"Step 12 complete: {record['backbone']} gate1=PASS")
        return 0

    print(f"Steps 13+ not implemented.", file=sys.stderr)
    return 1


def find_repo_root(start: Path) -> Path:
    from src.utils.config import find_repo_root as _find

    return _find(start)


if __name__ == "__main__":
    raise SystemExit(main())
