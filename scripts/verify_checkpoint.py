#!/usr/bin/env python3
"""
Verify a backbone checkpoint is reachable, cached, and loads to the declared shape.

Protocol Step 0/2 precondition: run this once per backbone before extraction, so that
authentication, gating, download and a wrong embed_dim all fail *here* — cheaply, in
seconds — rather than partway through a multi-day extraction job.

    python3 scripts/verify_checkpoint.py uni
    python3 scripts/verify_checkpoint.py --all

Uses the pipeline's own `create_timm_model`, deliberately: a parallel loader written
just for verification would drift from the production path and could pass while the
real extraction fails.

This script never handles credentials. If authentication is missing it says so and
exits; log in yourself with `hf auth login`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import find_repo_root, load_backbone_config  # noqa: E402

OK, BAD, WARN, SKIP = "  [ok]", "  [FAIL]", "  [warn]", "  [skip]"


class VerificationFailed(RuntimeError):
    """Raised with an actionable message; caught and printed, never a traceback."""


def _hub_repo_id(checkpoint: str) -> str | None:
    """'hf-hub:MahmoodLab/uni' -> 'MahmoodLab/uni'. None if not a hub reference."""
    if checkpoint and checkpoint.startswith("hf-hub:"):
        return checkpoint.split("hf-hub:", 1)[1]
    if checkpoint and checkpoint.count("/") == 1 and not Path(checkpoint).exists():
        return checkpoint
    return None


def check_auth() -> str | None:
    """Return the logged-in username, or None. Never reads or prints a token value."""
    from huggingface_hub import whoami

    try:
        return whoami().get("name")
    except Exception:
        return None


def check_gate(repo_id: str, *, authed: bool) -> None:
    """Confirm the repo's files are actually reachable, not just its metadata.

    Gated repos serve metadata to anyone. Only a file request proves access, so that
    is what this checks — otherwise a gated repo looks fine until download time.
    """
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import GatedRepoError

    info = HfApi().model_info(repo_id)
    gated = getattr(info, "gated", False)
    weight_files = [
        s.rfilename
        for s in (info.siblings or [])
        if s.rfilename.endswith((".bin", ".safetensors"))
    ]
    if not weight_files:
        raise VerificationFailed(f"{repo_id}: no weight files listed in the repo")

    print(f"{OK} repo reachable — gated={gated!r}, weights={weight_files[0]}")

    try:
        md = get_hf_file_metadata(hf_hub_url(repo_id, weight_files[0]))
    except GatedRepoError as exc:
        if not authed:
            raise VerificationFailed(
                f"{repo_id} is gated and you are not logged in.\n"
                "      Run:  hf auth login\n"
                "      Use a token from https://huggingface.co/settings/tokens"
            ) from exc
        raise VerificationFailed(
            f"{repo_id} is gated and your account has NOT been granted access.\n"
            f"      Accept the terms at https://huggingface.co/{repo_id}\n"
            "      UNI additionally requires your HF primary email to be your\n"
            "      institutional address — personal domains are rejected."
        ) from exc

    size = f"{md.size / 1e9:.2f} GB" if md.size else "unknown size"
    print(f"{OK} access granted — {weight_files[0]}, {size}")


def load_and_verify(cfg) -> None:
    """Load via the production path and check the embedding shape matches config."""
    import torch

    from src.backbone.loaders.timm_loader import create_timm_model

    model = create_timm_model(cfg)
    print(f"{OK} model constructed via create_timm_model()")

    # LayerScale trap: timm sets ls1 = nn.Identity() when init_values is absent, so
    # the weights load "successfully" minus their LayerScale gammas and the features
    # are silently wrong. configs/uni.yaml warns about this; assert it rather than
    # trusting the warning was read.
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) and hasattr(blocks[0], "ls1"):
        if isinstance(blocks[0].ls1, torch.nn.Identity):
            raise VerificationFailed(
                f"{cfg.name}: LayerScale is nn.Identity — init_values was not applied.\n"
                "      Features would be silently wrong. Check loader_kwargs.init_values."
            )
        print(f"{OK} LayerScale active (init_values applied)")

    size = getattr(model, "pretrained_cfg", {}).get("input_size", (3, 224, 224))
    with torch.no_grad():
        out = model(torch.zeros(1, *size))

    if out.ndim != 2:
        raise VerificationFailed(
            f"{cfg.name}: expected a 2-D pooled output, got shape {tuple(out.shape)}"
        )
    got = out.shape[1]
    if got != cfg.embed_dim:
        raise VerificationFailed(
            f"{cfg.name}: embed_dim mismatch — config says {cfg.embed_dim}, "
            f"forward pass gives {got}.\n"
            "      Fix the config; do not adjust downstream code to absorb it."
        )
    print(f"{OK} forward pass {tuple(size)} -> [1, {got}], matches config embed_dim")


def verify(config_path: Path) -> bool:
    print(f"\n=== {config_path.stem} ===")

    try:
        # Inside the try on purpose: a config with a null embed_dim or checkpoint is a
        # legitimate not-yet-pinned backbone, and --all must report it and carry on
        # rather than aborting the sweep on the first unpinned one.
        cfg = load_backbone_config(config_path)
        print(f"{OK} config loaded — {cfg.family}, embed_dim={cfg.embed_dim}")

        if not cfg.checkpoint:
            raise VerificationFailed(
                f"{cfg.name}: backbone.checkpoint is null — nothing to verify yet"
            )
        if not cfg.is_representation_resolved:
            raise VerificationFailed(
                f"{cfg.name}: representation.status is {cfg.representation_status!r}"
            )

        repo_id = _hub_repo_id(cfg.checkpoint)
        if repo_id:
            user = check_auth()
            print(f"{OK} authenticated as {user}" if user else f"{WARN} not logged in")
            check_gate(repo_id, authed=bool(user))
        else:
            print(f"{SKIP} not a Hugging Face hub reference: {cfg.checkpoint}")

        if cfg.loader != "timm":
            print(f"{SKIP} loader is {cfg.loader!r} — load check covers timm only")
            return True

        load_and_verify(cfg)

    except VerificationFailed as exc:
        print(f"{BAD} {exc}")
        return False
    except Exception as exc:  # unexpected — show the type, not a wall of traceback
        print(f"{BAD} {type(exc).__name__}: {str(exc)[:300]}")
        return False

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backbone", nargs="?", help="backbone name or path to its config")
    ap.add_argument("--all", action="store_true", help="verify every config")
    args = ap.parse_args()

    root = find_repo_root(Path(__file__))
    if args.all:
        configs = sorted(p for p in (root / "configs").glob("*.yaml") if not p.name.startswith("_"))
    elif args.backbone:
        p = Path(args.backbone)
        configs = [p if p.is_file() else root / "configs" / f"{args.backbone}.yaml"]
    else:
        ap.error("give a backbone name or --all")

    results = {c.stem: verify(c) for c in configs}

    print("\n" + "=" * 52)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
