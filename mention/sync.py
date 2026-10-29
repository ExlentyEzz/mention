"""Checkpoint hub + session lease (with retries and multi-file commits).

Backends:
  - "hf":   private Hugging Face model repo (recommended). Needs HF_TOKEN access.
  - "local": plain directory on disk (offline tests / local runs).

All HF writes go through a retry wrapper (429/5xx from HF Hub are transient)
and multi-file uploads are batched into a single commit to stay under rate
limits. Only the main (rank 0) process should call mutating hub methods;
trainers must guard with `accelerator.is_main_process`.
"""
from __future__ import annotations

import json
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


def _retry(fn, *args, retries: int = 6, base_delay: float = 2.0, **kwargs):
    """Call fn(*args, **kwargs), retrying on transient HTTP errors."""
    last = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - HF raises several exceptions
            import httpx
            from huggingface_hub.errors import HfHubHTTPError

            if isinstance(e, HfHubHTTPError):
                code = e.response.status_code if e.response is not None else None
                if code not in (429, 500, 502, 503, 504):
                    raise
            elif not isinstance(e, (httpx.HTTPError, ConnectionError, TimeoutError)):
                raise
            last = e
            delay = base_delay * (2 ** attempt)
            print(f"[sync] transient HF error ({e}); retry in {delay:.0f}s ...")
            time.sleep(delay)
    raise last


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
            _retry(
                self._api.create_repo,
                self.repo_id,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
        else:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            (self.local_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
            (self.local_dir / "lease").mkdir(parents=True, exist_ok=True)

    # -- low-level api -----------------------------------------------------

    def _exists(self, filename: str) -> bool:
        if self.backend == "hf":
            try:
                return _retry(self._api.file_exists, self.repo_id, filename)  # type: ignore[union-attr]
            except Exception:
                return False
        return (self.local_dir / filename).exists()

    def _read_json(self, filename: str) -> Optional[Dict[str, Any]]:
        if not self._exists(filename):
            return None
        if self.backend == "hf":
            path = _retry(  # type: ignore[union-attr]
                self._api.hf_hub_download,
                self.repo_id,
                filename,
                token=self.token,
            )
        else:
            path = self.local_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _upload_multiple(self, files: Dict[str, str]) -> None:
        """Upload {path_in_repo: local_path} in one commit (HF only)."""
        if self.backend == "local":
            for path_in_repo, local_path in files.items():
                dst = self.local_dir / path_in_repo
                dst.parent.mkdir(parents=True, exist_ok=True)
                if Path(local_path).resolve() != dst.resolve():
                    shutil.copy(local_path, dst)
            return
        from huggingface_hub import UploadOperation

        ops = [
            UploadOperation(path_or_fileobj=local_path, path_in_repo=path_in_repo)
            for path_in_repo, local_path in files.items()
        ]
        _retry(
            self._api.create_commit,  # type: ignore[union-attr]
            self.repo_id,
            operations=ops,
            commit_message="mention: sync checkpoint",
        )

    def _write_json(self, filename: str, obj: Dict[str, Any]) -> None:
        tmp = self.local_dir / filename
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        self._upload_multiple({filename: str(tmp)})

    def _delete(self, filename: str) -> None:
        if self.backend == "hf":
            try:
                _retry(self._api.delete_file, filename, self.repo_id)  # type: ignore[union-attr]
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
        if not self._exists(MODEL_FILENAME):
            print("[sync] model weights missing in hub; returning no checkpoint")
            return None
        if self.backend == "hf":
            _retry(  # type: ignore[union-attr]
                self._api.hf_hub_download,
                self.repo_id,
                MODEL_FILENAME,
                local_dir=str(workdir),
                token=self.token,
            )
        else:
            shutil.copy(self.local_dir / MODEL_FILENAME, workdir / MODEL_FILENAME)
        # tokenizer is best-effort (required only for the first launch)
        try:
            if self.backend == "hf":
                _retry(  # type: ignore[union-attr]
                    self._api.hf_hub_download,
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
        files: Dict[str, str] = {}
        meta_path = workdir / META_FILENAME
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        files[META_FILENAME] = str(meta_path)
        files[MODEL_FILENAME] = str(workdir / MODEL_FILENAME)
        if include_tokenizer and (workdir / TOKENIZER_FILENAME).exists():
            files[TOKENIZER_FILENAME] = str(workdir / TOKENIZER_FILENAME)
        self._upload_multiple(files)

    def upload_seed_files(self, workdir: str | Path) -> None:
        """Upload tokenizer + a bootstrap meta.json once (called by seed script)."""
        workdir = Path(workdir)
        files: Dict[str, str] = {}
        if (workdir / TOKENIZER_FILENAME).exists():
            files[TOKENIZER_FILENAME] = str(workdir / TOKENIZER_FILENAME)
        meta = workdir / META_FILENAME
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(
            json.dumps(
                {"step": 0, "tokens": 0, "seed": True, "created_at": now_iso()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        files[META_FILENAME] = str(meta)
        self._upload_multiple(files)

    def pull_tokenizer_if_missing(self, workdir: str | Path) -> bool:
        """Best-effort download of tokenizer even when no checkpoint exists yet."""
        workdir = Path(workdir)
        if (workdir / TOKENIZER_FILENAME).exists():
            return True
        if not self._exists(TOKENIZER_FILENAME):
            return False
        try:
            if self.backend == "hf":
                _retry(  # type: ignore[union-attr]
                    self._api.hf_hub_download,
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
