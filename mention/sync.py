"""Checkpoint hub + session lease.

Backends:
  - "hf":   private Hugging Face model repo (recommended). Needs HF_TOKEN access.
  - "local": plain directory on disk (offline tests / local runs).

Repo layout:
  checkpoint/model.safetensors   latest model weights (fp32)
  checkpoint/meta.json           {"step": int, "tokens": int, "lr": float, ...}
  lease/lease.json               {"owner": str, "started_at", "heartbeat_at", "expires_at"}

Lease semantics:
  Only one lane trains at a time. A lane tries to acquire the lease on start;
  if it is alive (heartbeat within ttl), it waits; otherwise it takes over.
  The owner must heartbeat periodically (default every 45 min, ttl 90 min)
  and release on clean exit.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import now_iso

LEASE_FILENAME = "lease/lease.json"
META_FILENAME = "checkpoint/meta.json"
MODEL_FILENAME = "checkpoint/model.safetensors"
TOKENIZER_FILENAME = "checkpoint/tokenizer.json"
CONFIG_FILENAME = "checkpoint/config.yaml"


class CheckpointHub:
    def __init__(
        self,
        backend: str = "hf",
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        local_dir: Optional[str] = None,
        timeout_s: int = 600,
    ):
        self.backend = backend
        self.repo_id = repo_id
        self.token = token
        self.local_dir = Path(local_dir) if local_dir else Path("hub_local")
        self.timeout_s = timeout_s
        self._api = None

    # -- setup -------------------------------------------------------------

    def ensure_ready(self) -> None:
        if self.backend == "hf":
            if not self.repo_id:
                raise ValueError("repo_id is required for hf backend")
            if not self.token:
                raise ValueError("HF token is required (set HF_TOKEN / lane env var)")
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
            self._api.create_repo(
                self.repo_id, repo_type="model", private=True, exist_ok=True
            )
        else:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            (self.local_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
            (self.local_dir / "lease").mkdir(parents=True, exist_ok=True)

    # -- low-level api -----------------------------------------------------

    def _exists(self, filename: str) -> bool:
        if self.backend == "hf":
            try:
                self._api.file_exists(self.repo_id, filename)  # type: ignore[union-attr]
            except Exception:
                return False
        else:
            return (self.local_dir / filename).exists()

    def _read_json(self, filename: str) -> Optional[Dict[str, Any]]:
        if not self._exists(filename):
            return None
        if self.backend == "hf":
            path = self._api.hf_hub_download(  # type: ignore[union-attr]
                self.repo_id, filename, token=self.token
            )
        else:
            path = self.local_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, filename: str, obj: Dict[str, Any]) -> None:
        tmp = self.local_dir / filename
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.backend == "hf":
            self._api.upload_file(  # type: ignore[union-attr]
                path_or_fileobj=str(tmp),
                path_in_repo=filename,
                repo_id=self.repo_id,
            )

    def _delete(self, filename: str) -> None:
        if self.backend == "hf":
            try:
                self._api.delete_file(filename, self.repo_id)  # type: ignore[union-attr]
            except Exception:
                pass
        else:
            p = self.local_dir / filename
            if p.exists():
                p.unlink()

    # -- checkpoint pull/push ----------------------------------------------

    def pull_latest(self, workdir: str | Path) -> Optional[Dict[str, Any]]:
        """Download latest checkpoint into workdir, return meta dict or None.

        Tolerates a missing tokenizer (legacy checkpoints / smoke runs).
        """
        workdir = Path(workdir)
        meta = self._read_json(META_FILENAME)
        if meta is None:
            return None
        try:
            if self.backend == "hf":
                self._api.hf_hub_download(  # type: ignore[union-attr]
                    self.repo_id,
                    MODEL_FILENAME,
                    local_dir=str(workdir),
                    token=self.token,
                )
            else:
                shutil.copy(self.local_dir / MODEL_FILENAME, workdir / MODEL_FILENAME)
        except FileNotFoundError:
            print("[sync] model weights missing in hub; returning no checkpoint")
            return None
        # tokenizer is best-effort (required only for the first launch)
        try:
            if self.backend == "hf":
                self._api.hf_hub_download(  # type: ignore[union-attr]
                    self.repo_id,
                    TOKENIZER_FILENAME,
                    local_dir=str(workdir),
                    token=self.token,
                )
            else:
                shutil.copy(
                    self.local_dir / TOKENIZER_FILENAME,
                    workdir / TOKENIZER_FILENAME,
                )
        except FileNotFoundError:
            print("[sync] tokenizer.json not in hub (OK if already local)")
        return meta

    def push_checkpoint(
        self,
        workdir: str | Path,
        meta: Dict[str, Any],
        include_tokenizer: bool = True,
    ) -> None:
        workdir = Path(workdir)
        self._write_json(META_FILENAME, meta)
        if self.backend == "hf":
            self._api.upload_file(  # type: ignore[union-attr]
                path_or_fileobj=str(workdir / MODEL_FILENAME),
                path_in_repo=MODEL_FILENAME,
                repo_id=self.repo_id,
            )
            if include_tokenizer and (workdir / TOKENIZER_FILENAME).exists():
                self._api.upload_file(  # type: ignore[union-attr]
                    path_or_fileobj=str(workdir / TOKENIZER_FILENAME),
                    path_in_repo=TOKENIZER_FILENAME,
                    repo_id=self.repo_id,
                )
        else:
            shutil.copy(workdir / MODEL_FILENAME, self.local_dir / MODEL_FILENAME)
            if include_tokenizer and (workdir / TOKENIZER_FILENAME).exists():
                shutil.copy(
                    workdir / TOKENIZER_FILENAME, self.local_dir / TOKENIZER_FILENAME
                )

    def upload_seed_files(self, workdir: str | Path) -> None:
        """Upload tokenizer + a bootstrap meta.json once (called by seed script).

        meta.json is needed so that pull_latest() (which keys off meta) will
        also fetch the tokenizer on the very first training run.
        """
        workdir = Path(workdir)
        if (workdir / TOKENIZER_FILENAME).exists():
            if self.backend == "hf":
                self._api.upload_file(  # type: ignore[union-attr]
                    path_or_fileobj=str(workdir / TOKENIZER_FILENAME),
                    path_in_repo=TOKENIZER_FILENAME,
                    repo_id=self.repo_id,
                )
            else:
                shutil.copy(
                    workdir / TOKENIZER_FILENAME, self.local_dir / TOKENIZER_FILENAME
                )
        self._write_json(
            META_FILENAME,
            {"step": 0, "tokens": 0, "seed": True, "created_at": now_iso()},
        )

    def pull_tokenizer_if_missing(self, workdir: str | Path) -> bool:
        """Best-effort download of tokenizer even when no checkpoint exists yet."""
        workdir = Path(workdir)
        if (workdir / TOKENIZER_FILENAME).exists():
            return True
        if not self._exists(TOKENIZER_FILENAME):
            return False
        try:
            if self.backend == "hf":
                self._api.hf_hub_download(  # type: ignore[union-attr]
                    self.repo_id,
                    TOKENIZER_FILENAME,
                    local_dir=str(workdir),
                    token=self.token,
                )
            else:
                shutil.copy(
                    self.local_dir / TOKENIZER_FILENAME,
                    workdir / TOKENIZER_FILENAME,
                )
            return True
        except FileNotFoundError:
            return False

    # -- lease -------------------------------------------------------------

    def get_lease(self) -> Optional[Dict[str, Any]]:
        return self._read_json(LEASE_FILENAME)

    def acquire_lease(
        self,
        owner: str,
        ttl_min: int = 90,
        poll_s: int = 30,
        max_wait_min: int = 8,
        force: bool = False,
    ) -> bool:
        """Try to take the lease. Returns True when we own it.

        force=True steals the lease immediately (use manually when a session
        died without releasing).
        """
        if force:
            self._steal(owner, ttl_min)
            return True

        deadline = time.time() + max_wait_min * 60
        while True:
            lease = self.get_lease()
            if lease is None:
                self._steal(owner, ttl_min)
                back = self.get_lease()
                if back and back.get("owner") == owner:
                    return True
            else:
                hb = float(lease.get("heartbeat_at", 0))
                if time.time() - hb < ttl_min * 60:
                    if time.time() > deadline:
                        print(
                            f"[lease] still owned by {lease.get('owner')} "
                            f"(heartbeat {hb:.0f}); giving up. Re-run with --force "
                            "if that session is dead."
                        )
                        return False
                    if int(hb) % poll_s == 0:  # avoid spamming every poll
                        print(
                            f"[lease] waiting on {lease.get('owner')} "
                            f"({int(deadline - time.time())}s left) ..."
                        )
                    time.sleep(poll_s)
                else:
                    print(f"[lease] stale lease from {lease.get('owner')}; taking over")
                    self._steal(owner, ttl_min)
                    back = self.get_lease()
                    if back and back.get("owner") == owner:
                        return True

    def _steal(self, owner: str, ttl_min: int) -> None:
        lease = self.get_lease()
        self._write_json(
            LEASE_FILENAME,
            {
                "owner": owner,
                "started_at": now_iso(),
                "heartbeat_at": time.time(),
                "expires_at": time.time() + ttl_min * 60,
                "step": lease.get("step", 0) if lease else 0,
            },
        )

    def heartbeat(self, owner: str, ttl_min: int = 90) -> None:
        lease = self.get_lease()
        if lease and lease.get("owner") == owner:
            lease["heartbeat_at"] = time.time()
            lease["expires_at"] = time.time() + ttl_min * 60
            self._write_json(LEASE_FILENAME, lease)

    def release(self, owner: str) -> None:
        lease = self.get_lease()
        if lease and lease.get("owner") == owner:
            self._delete(LEASE_FILENAME)


class HeartbeatThread(threading.Thread):
    """Daemon that refreshes the lease every interval seconds."""

    def __init__(self, hub: CheckpointHub, owner: str, interval_s: int = 2700):
        super().__init__(daemon=True)
        self.hub = hub
        self.owner = owner
        self.interval_s = interval_s
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.interval_s)
            if not self._stop.is_set():
                self.hub.heartbeat(self.owner)

    def stop(self) -> None:
        self._stop.set()
