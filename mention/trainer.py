"""Pretraining loop for the masked-diffusion model.

Key features:
- MDLM objective: per-sample random mask ratio t ~ U(0,1); predict masked tokens
  (cross-entropy), plus a small share of "origin" (unmasked) examples.
- FP16 mixed precision (T4/P100 have no bf16), PyTorch SDPA, grad accumulation.
- Periodic checkpoint push to the hub (every push_every_min) and graceful stop
  at a wall-clock budget (--budget-min), so 12h Kaggle sessions never lose work.
- Session lease heartbeat runs in a background thread.
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator

from .data import StreamingLMCorpus, TinySmokeCorpus, make_dataloader
from .model import MaskedDiffusionTransformer, ModelConfig
from .sync import CheckpointHub, HeartbeatThread
from .utils import AvgMeter, count_params, estimate_tok_per_sec, print_rank0, set_seed


def _am_i_main() -> bool:
    """True on the global rank-0 process (works with `accelerate launch`)."""
    try:
        return os.environ.get("LOCAL_RANK", "0") in ("0", "")
    except Exception:
        return True


class TrainConfig:
    def __init__(self, d: Dict[str, Any]):
        t = d.get("train", {})
        self.max_steps = int(t.get("max_steps", 12000))
        self.learning_rate = float(t.get("learning_rate", 6e-4))
        self.min_lr_ratio = float(t.get("min_lr_ratio", 0.1))
        self.warmup_steps = int(t.get("warmup_steps", 400))
        self.weight_decay = float(t.get("weight_decay", 0.1))
        self.beta1 = float(t.get("beta1", 0.9))
        self.beta2 = float(t.get("beta2", 0.95))
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.micro_batch_size = int(t.get("micro_batch_size", 8))
        self.grad_accum = int(t.get("grad_accum", 4))
        self.fp16 = bool(t.get("fp16", True))
        self.origin_prob = float(t.get("origin_prob", 0.015))
        self.log_every_steps = int(t.get("log_every_steps", 20))
        self.save_every_steps = int(t.get("save_every_steps", 500))
        self.push_every_min = int(t.get("push_every_min", 120))
        self.en_prob = float(t.get("en_prob", 0.55))
        self.tokenizer_samples = int(t.get("tokenizer_samples", 60000))
        self.tokenizer_vocab = int(d.get("data", {}).get("tokenizer_vocab", 32768))
        self.seed = int(t.get("seed", 42))
        self.smoke = bool(d.get("smoke", False))
        self.seq_len = int(d.get("data", {}).get("seq_len", 2048))
        if self.smoke:
            self.seq_len = min(self.seq_len, 256)
            self.micro_batch_size = min(self.micro_batch_size, 2)
        self.gradient_checkpointing = bool(t.get("gradient_checkpointing", True))

    @property
    def tokens_per_opt_step(self) -> int:
        return self.micro_batch_size * self.grad_accum * self.seq_len


def _lr_at_step(lr: float, min_lr: float, step: int, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_smoke_tokenizer():
    """Tiny BPE trained offline from toy lines — for smoke tests only."""
    from .tokenizer import train_bpe
    from .data import TinySmokeCorpus

    tmp = Path("runs") / "smoke_lines.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(TinySmokeCorpus.LINES), encoding="utf-8")
    out = Path("runs") / "smoke_tokenizer.json"
    train_bpe([tmp], vocab_size=512, out_path=out)
    return out


def run_pretrain(
    cfg: TrainConfig,
    hub: CheckpointHub,
    lane_name: str,
    workdir: str | Path,
    budget_min: Optional[int] = None,
    max_steps_override: Optional[int] = None,
    force_lease: bool = False,
) -> Dict[str, Any]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    max_steps = max_steps_override or cfg.max_steps

    # ---- model config ----------------------------------------------------
    from .model import ModelConfig  # noqa: F401 (re-import for clarity)

    # tokenizer: pull from hub, or bootstrap-train it right here on the lane.
    # Only the main rank trains/uploadsa; other ranks wait for the file.
    tok_path = workdir / "checkpoint/tokenizer.json"
    if not tok_path.exists():
        if cfg.smoke:
            tok_path = make_smoke_tokenizer()
        else:
            have = hub.pull_tokenizer_if_missing(workdir)
            if not have:
                if _am_i_main():
                    print_rank0(
                        f"[{lane_name}] no tokenizer in hub — bootstrapping one "
                        f"(vocab {cfg.tokenizer_vocab}, "
                        f"{cfg.tokenizer_samples} samples/source) ..."
                    )
                    from .data import iter_raw_texts
                    from .tokenizer import train_bpe_from_iterators

                    en = iter_raw_texts(
                        "HuggingFaceFW/fineweb-edu", None, cfg.tokenizer_samples
                    )
                    ru = iter_raw_texts(
                        "HuggingFaceFW/fineweb-2", "rus_Cyrl", cfg.tokenizer_samples
                    )
                    tok_path.parent.mkdir(parents=True, exist_ok=True)
                    train_bpe_from_iterators(
                        [en, ru],
                        vocab_size=cfg.tokenizer_vocab,
                        out_path=tok_path,
                        min_frequency=2,
                    )
                    hub.upload_seed_files(workdir)
                    print_rank0(f"[{lane_name}] bootstrap tokenizer saved and uploaded")
                else:
                    # non-main ranks wait until main created the tokenizer file
                    deadline = time.monotonic() + 900
                    while not tok_path.exists():
                        time.sleep(5)
                        if time.monotonic() > deadline:
                            break
            if not tok_path.exists():
                raise FileNotFoundError(
                    f"tokenizer not found at {tok_path} (and could not be created)"
                )
    from .tokenizer import load_tokenizer, mask_id

    tok = load_tokenizer(tok_path)
    mask_token_id = mask_id(tok)

    mcfg_dict = {"vocab_size": tok.get_vocab_size(), "max_seq_len": cfg.seq_len}
    # merge with model section of yaml (already applied by caller)
    # caller passes full config via args; here we attach tokenizer vocab
    # -> we accept model dict from caller inside cfg? keep minimal: read from
    # a saved config copy in workdir if present.
    cfg_path = workdir / "config.yaml"
    if cfg_path.exists():
        import yaml

        full = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        mcfg = ModelConfig.from_dict(full.get("model", {}))
        mcfg.vocab_size = tok.get_vocab_size()
    else:
        mcfg = ModelConfig(vocab_size=tok.get_vocab_size(), max_seq_len=cfg.seq_len)

    model = MaskedDiffusionTransformer(mcfg)
    n_params = count_params(model)
    print_rank0(
        f"[{lane_name}] model parameters: {n_params/1e6:.1f}M, "
        f"vocab={mcfg.vocab_size}, seq_len={cfg.seq_len}"
    )

    # ---- data ------------------------------------------------------------
    if cfg.smoke:
        corpus = TinySmokeCorpus(tok, cfg.seq_len)
    else:
        corpus = StreamingLMCorpus(
            tokenizer=tok,
            seq_len=cfg.seq_len,
            en_prob=cfg.en_prob,
        )
    loader = make_dataloader(corpus, cfg.micro_batch_size)

    # ---- optimizer / accelerator ----------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    use_fp16 = cfg.fp16 and torch.cuda.is_available()
    accelerator = Accelerator(
        mixed_precision="fp16" if use_fp16 else "no",
        gradient_accumulation_steps=cfg.grad_accum,
        log_with=None,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # ---- resume ----------------------------------------------------------
    start_step = 0
    meta = hub.pull_latest(workdir)
    if meta is not None and (workdir / "checkpoint/model.safetensors").exists():
        import safetensors.torch as st

        state = st.load_file(str(workdir / "checkpoint/model.safetensors"))
        unwrapped = accelerator.unwrap_model(model)
        # strict=False: lm_head.weight is tied to wte.weight and not stored
        unwrapped.load_state_dict(state, strict=False)
        start_step = int(meta.get("step", 0))
        print_rank0(f"[{lane_name}] resumed from step {start_step}")

    # ---- lease -----------------------------------------------------------
    # Only the main rank talks to the hub; the outcome is shared via a marker
    # file on the shared filesystem (workdir is shared across ranks).
    ok = True
    if accelerator.is_main_process:
        ok = hub.acquire_lease(
            lane_name, ttl_min=90, max_wait_min=8, force=force_lease
        )
        marker = workdir / ("lease_ok" if ok else "lease_busy")
        marker.touch(exist_ok=True)
    else:
        deadline = time.monotonic() + 120
        while not (workdir / "lease_ok").exists() and not (
            workdir / "lease_busy"
        ).exists():
            if time.monotonic() > deadline:
                break
            time.sleep(1)
        ok = (workdir / "lease_ok").exists()
    if not ok:
        print_rank0(
            f"[{lane_name}] lease busy (another lane trains). Exiting without training."
        )
        return {"status": "lease-busy", "step": start_step}

    hb: HeartbeatThread | None = None
    if accelerator.is_main_process:
        hb = HeartbeatThread(hub, lane_name, interval_s=2700)  # 45 min
        hb.start()

    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    loss_meter = AvgMeter()
    tok_meter = AvgMeter()
    step = start_step
    tokens_total = int(meta.get("tokens", 0)) if meta else 0
    t0 = time.monotonic()
    t_last = t0
    push_deadline = t0 + cfg.push_every_min * 60
    end_deadline = t0 + budget_min * 60 if budget_min else None
    loss_path = workdir / "loss.csv"
    with open(loss_path, "a", newline="", encoding="utf-8") as f:
        if step == 0:
            w = csv.writer(f)
            w.writerow(["step", "tokens", "loss", "tokens_per_s", "lr"])

    def save_and_push(final: bool = False):
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(model)
        state = {k: v.detach().float().cpu() for k, v in unwrapped.state_dict().items()}
        # lm_head.weight is tied to wte.weight (shared memory) - save once
        state.pop("lm_head.weight", None)
        ck_dir = workdir / "checkpoint"
        ck_dir.mkdir(parents=True, exist_ok=True)
        import safetensors.torch as st

        st.save_file(state, str(ck_dir / "model.safetensors"))
        meta_out = {
            "step": step,
            "tokens": tokens_total,
            "lr": lr_now,
            "lane": lane_name,
            "saved_at": time.time(),
            "final": final,
        }
        (ck_dir / "meta.json").write_text(
            __import__("json").dumps(meta_out, indent=2), encoding="utf-8"
        )
        hub.push_checkpoint(workdir, meta_out, include_tokenizer=(step == start_step or final))
        print_rank0(f"[{lane_name}] pushed checkpoint at step {step}")

    try:
        loader_iter = iter(loader)
        while step < max_steps:
            # --- LR ------------------------------------------------------
            lr_now = _lr_at_step(
                cfg.learning_rate, min_lr, step, cfg.warmup_steps, max_steps
            )
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            # --- forward/backward with accumulation ----------------------
            with accelerator.accumulate(model):
                batch = next(loader_iter)
                ids = batch["input_ids"]
                B, L = ids.shape
                r = torch.rand(B, device=ids.device)
                mask = torch.rand(B, L, device=ids.device) < r[:, None]
                origin = torch.rand(B, device=ids.device) < cfg.origin_prob
                mask[origin] = False
                masked = mask.any()
                with accelerator.autocast():
                    logits = model(ids)
                losses = []
                if masked:
                    flat_logits = logits.view(-1, logits.size(-1))[mask.view(-1)]
                    flat_tgt = ids.view(-1)[mask.view(-1)]
                    losses.append(
                        torch.nn.functional.cross_entropy(flat_logits, flat_tgt)
                    )
                else:
                    # rare degenerate batch (no masked positions): just zero
                    losses.append(logits.float().sum() * 0.0)
                loss = torch.stack(losses).mean()
                accelerator.backward(loss)

            # optimizer step happens inside accumulate block automatically
            # (accelerate steps once grad_accum reached); compute loss log after
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                tokens_total += B * L * accelerator.num_processes
                loss_meter.add(loss.item())
                tok_meter.add(B * L * accelerator.num_processes, 1)

            # --- logging --------------------------------------------------
            if step % cfg.log_every_steps == 0 and step > start_step:
                now = time.monotonic()
                dt = now - t_last
                t_last = now
                tps = tok_meter.value / max(dt, 1e-6)
                tok_meter.reset()
                print_rank0(
                    f"[{lane_name}] step {step}/{max_steps} | loss {loss_meter.value:.4f} "
                    f"| {tps/1e3:.1f}k tok/s | lr {lr_now:.2e} | "
                    f"wall {int(now-t0)}s"
                )
                with open(loss_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([step, tokens_total, loss_meter.value, tps, lr_now])
                loss_meter.reset()

            # --- periodic save/push --------------------------------------
            now = time.monotonic()
            if now >= push_deadline:
                save_and_push()
                push_deadline = now + cfg.push_every_min * 60
            if end_deadline is not None and now >= end_deadline:
                print_rank0(f"[{lane_name}] budget reached, finalizing...")
                save_and_push(final=True)
                break

        # normal end of max_steps: always leave a final checkpoint behind
        if step >= max_steps and step > start_step:
            print_rank0(f"[{lane_name}] max_steps reached, finalizing...")
            save_and_push(final=True)
    except KeyboardInterrupt:
        print_rank0(f"[{lane_name}] interrupted, saving...")
        save_and_push(final=True)
    finally:
        if hb is not None:
            hb.stop()
        if accelerator.is_main_process:
            hub.release(lane_name)

    wall = time.monotonic() - t0
    tps = estimate_tok_per_sec(step - start_step, cfg.tokens_per_opt_step, wall)
    print_rank0(
        f"[{lane_name}] done: {step} steps, {tokens_total/1e9:.2f}B tokens, "
        f"{wall/3600:.1f}h, {tps/1e3:.1f}k tok/s"
    )
    return {"status": "ok", "step": step, "tokens": tokens_total, "wall_s": wall}
