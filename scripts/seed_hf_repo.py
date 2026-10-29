#!/usr/bin/env python3
"""Create the private HF repo and upload the tokenizer + configs.

Run once after training the tokenizer:

  python scripts/seed_hf_repo.py \
      --repo YOUR_HF_USERNAME/mention-ckpt \
      --artifacts artifacts/tokenizer \
      --config configs/pretrain_300m.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mention.sync import CheckpointHub  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--artifacts", default="artifacts/tokenizer")
    p.add_argument("--config", default=None, help="config yaml to store alongside")
    p.add_argument("--token", default=None)
    p.add_argument("--local-dir", default=None, help="offline: just copy to a dir")
    args = p.parse_args()

    artifacts = Path(args.artifacts)
    if not (artifacts / "tokenizer.json").exists():
        sys.exit("tokenizer.json not found in --artifacts; run train_tokenizer.py first")

    hub = CheckpointHub(
        backend="local" if args.local_dir else "hf",
        repo_id=None if args.local_dir else args.repo,
        token=args.token,
        local_dir=args.local_dir,
    )
    hub.ensure_ready()

    workdir = artifacts / "_seed"
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(artifacts / "tokenizer.json", workdir / "checkpoint/tokenizer.json")
    if args.config:
        shutil.copy(args.config, workdir / "config.yaml")
    hub.upload_seed_files(workdir)
    print(f"[seed] uploaded tokenizer to {args.repo or args.local_dir}")


if __name__ == "__main__":
    main()
