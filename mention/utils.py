"""Shared helpers: config loading, seeding, metrics, tiny CLI utilities."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_tok_per_sec(steps: int, tokens_per_batch: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return steps * tokens_per_batch / seconds


def print_rank0(msg: str) -> None:
    try:
        import os

        rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        rank = 0
    if rank == 0:
        print(msg, flush=True)


class AvgMeter:
    """Running average with counter."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._n = 0

    def add(self, value: float, n: int = 1) -> None:
        self._sum += value * n
        self._n += n

    @property
    def value(self) -> float:
        return self._sum / self._n if self._n else 0.0


def now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
