#!/usr/bin/env python3
"""Universal lane entrypoint.

Trains the model on whatever GPU environment we are in (Kaggle notebook,
Colab, or a local machine). The same code is used everywhere; only the
--lane profile and --repo change.

Usage:
  python scripts/run_lane.py \
      --config configs/pilot_40m.yaml \
      --lane kaggle-1 \
      --repo YOUR_HF_USERNAME/mention-ckpt

  # offline smoke test (no internet, no HF):
  python scripts/run_lane.py \
      --config configs/pilot_40m.yaml \
      --lane smoke \
      --local-dir runs/hub_smoke \
      --smoke --steps 20
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mention.sync import CheckpointHub  # noqa: E402
from mention.trainer import TrainConfig, run_pretrain  # noqa: E402
from mention.utils import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Mention diffusion LM lane")
    p.add_argument("--config", required=True, help="path to model/config yaml")
    p.add_argument("--lane", default="kaggle-1", help="lane name from lanes.yaml")
    p.add_argument(
        "--repo",
        default=None,
        help="HF repo_id like 'user/mention-ckpt' (required unless --local-dir)",
    )
    p.add_argument("--local-dir", default=None, help="use local dir as hub (offline)")
    p.add_argument("--token", default=None, help="HF token override")
    p.add_argument("--steps", type=int, default=None, help="override max_steps")
    p.add_argument("--budget-min", type=int, default=None, help="override time budget")
    p.add_argument(
        "--push-every-min", type=int, default=None, help="override push interval"
    )
    p.add_argument("--workdir", default=None, help="working dir for checkpoints")
    p.add_argument("--smoke", action="store_true", help="tiny offline run")
    p.add_argument(
        "--force",
        action="store_true",
        help="steal the lease immediately (use if the previous session died)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = load_yaml(args.config)

    lanes = load_yaml(ROOT / "configs" / "lanes.yaml").get("lanes", {})
    lane = lanes.get(args.lane, {"type": "unknown", "hf_token_env": "HF_TOKEN"})

    # --- hub --------------------------------------------------------------
    if args.local_dir:
        hub = CheckpointHub(backend="local", local_dir=args.local_dir)
        budget_min = args.budget_min or 10
        repo_id = None
    else:
        if not args.repo:
            print("Provide --repo (HF repo_id) or --local-dir for offline mode.")
            sys.exit(2)
        token = args.token or os.environ.get(lane.get("hf_token_env", "HF_TOKEN")) or os.environ.get("HF_TOKEN")
        if not token:
            print(
                f"HF token not found: set env {lane.get('hf_token_env', 'HF_TOKEN')} "
                "or pass --token"
            )
            sys.exit(1)
        hub = CheckpointHub(backend="hf", repo_id=args.repo, token=token)
        budget_min = args.budget_min or int(lane.get("budget_min", 660))
    hub.ensure_ready()

    # --- workdir ----------------------------------------------------------
    workdir = Path(args.workdir) if args.workdir else ROOT / "runs" / args.lane
    workdir.mkdir(parents=True, exist_ok=True)
    # copy config into workdir so the trainer can re-read it on resume
    shutil.copy(args.config, workdir / "config.yaml")

    cfg = TrainConfig(cfg_dict)
    cfg.smoke = args.smoke
    if cfg.smoke:
        cfg.seq_len = min(cfg.seq_len, 256)
        cfg.micro_batch_size = min(cfg.micro_batch_size, 2)
        cfg.grad_accum = 1
    # allow data-selection via env (used by notebooks to tweak mix)
    cfg.en_prob = float(os.environ.get("MENTION_EN_PROB", "0.55"))

    run_pretrain(
        cfg,
        hub=hub,
        lane_name=args.lane,
        workdir=workdir,
        budget_min=budget_min,
        max_steps_override=args.steps,
        force_lease=args.force,
    )


if __name__ == "__main__":
    main()
