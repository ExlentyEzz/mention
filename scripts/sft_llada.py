#!/usr/bin/env python3
"""SFT of a LLaDA-family diffusion LM (e.g. LLaDA-MoE-7B-A1B) into a custom
"Mention-1"-style model.

Objective: masked-diffusion SFT. We keep the prompt tokens unmasked, mask a
random fraction of the response tokens (ratio ~ U(0.05, 0.95)), and train the
model to predict the masked response tokens. This is the standard conditional
diffusion training objective for LLaDA-style models.

Runs on a single GPU (Kaggle T4 works with QLoRA + small batch). The full 7B
model is loaded in 4-bit and fine-tuned with LoRA adapters; at the end you can
optionally merge and push the finished model to the Hugging Face Hub.

Example:
  python scripts/sft_llada.py \
      --base-model inclusionAI/LLaDA-MoE-7B-A1B-Base \
      --dataset yahma/alpaca-cleaned \
      --steps 1500 --batch-size 1 --grad-accum 8 \
      --lora-r 16 --lora-alpha 32 --load-in-4bit \
      --out runs/sft-mention-1 \
      --push-repo YOUR_HF_USERNAME/mention-1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA/QLoRA SFT of LLaDA-family")
    p.add_argument("--base-model", default="inclusionAI/LLaDA-MoE-7B-A1B-Base")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--split", default="train")
    p.add_argument("--subsample", type=int, default=0, help="0 = use all (streaming)")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--compute-dtype", default="auto", help="auto|fp16|bf16|fp32")
    # output
    p.add_argument("--out", default="runs/sft-mention-1")
    p.add_argument("--push-repo", default=None, help="HF repo to push merged model")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-sample-every", type=int, default=250)
    p.add_argument("--eval-prompt", default="Write a Python function to compute the Fibonacci numbers up to n.")
    return p.parse_args()


def pick_compute_dtype(requested: str) -> torch.dtype:
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 \
            else torch.float16
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[requested]


def discover_target_modules(model: nn.Module) -> List[str]:
    """Auto-detect linear module names to apply LoRA to."""
    names: set[str] = set()
    suffixes = ("_proj", "_out")
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            leaf = n.split(".")[-1]
            if leaf.endswith(suffixes):
                names.add(leaf)
    if not names:
        # fallback: any linear not in norms/embedding
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                leaf = n.split(".")[-1]
                if leaf not in ("lm_head",) and m.weight.shape[0] < 100000:
                    names.add(leaf)
    return sorted(names, key=lambda s: (s not in ("q_proj", "k_proj", "v_proj", "attn_out"), s))


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    compute = pick_compute_dtype(args.compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    load_kwargs: dict = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute,
        )
        load_kwargs["quantization_config"] = bnb
    else:
        load_kwargs["torch_dtype"] = compute
        load_kwargs["device_map"] = "auto"

    # Compatibility shim: some transformers versions lack the "default" RoPE
    # registry entry that LLaDA-MoE remote code expects.
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS:
            from transformers.modeling_rope_utils import _compute_default_rope_parameters

            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except Exception:
        pass
    model = AutoModel.from_pretrained(args.base_model, **load_kwargs)
    if args.load_in_4bit:
        model = model.to("cuda")
    model.train()
    return model, tokenizer, compute


def build_lora(model: nn.Module, args, compute: torch.dtype):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = discover_target_modules(model)
    print(f"[sft] LoRA target modules: {targets}")
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def make_example_iterator(args, tokenizer) -> Iterator[dict]:
    """Yield {prompt, output} dicts from a HF instruction dataset (streaming)."""
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=(args.subsample == 0))
    count = 0
    for ex in ds:
        # support alpaca / dolly / sharegpt-ish shapes
        instruction = ex.get("instruction") or ex.get("prompt") or ex.get("input") or ""
        inp = ex.get("input") or ""
        out = ex.get("output") or ex.get("response") or ex.get("answer") or ""
        if isinstance(instruction, list):
            # maybe sharegpt-like [{role,content},...]
            try:
                instruction = " ".join(m.get("content", "") for m in instruction)
            except Exception:
                instruction = ""
        if isinstance(out, list):
            try:
                out = " ".join(m.get("content", "") for m in out)
            except Exception:
                out = ""
        if not out:
            continue
        if inp and inp != instruction:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        yield {"prompt": prompt, "output": out}
        count += 1
        if args.subsample and count >= args.subsample:
            break


def encode_example(tokenizer, ex: dict, max_seq_len: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (ids, mask) where mask marks positions to predict (response only)."""
    prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
    tgt_ids = tokenizer(ex["output"], add_special_tokens=False).input_ids
    # cap lengths
    if len(prompt_ids) + len(tgt_ids) > max_seq_len - 2:
        budget = max_seq_len - 2 - len(prompt_ids)
        if budget <= 0:
            return None
        tgt_ids = tgt_ids[:budget]
    ids = prompt_ids + tgt_ids + [tokenizer.eos_token_id]
    seq_len = len(ids)
    n_resp = len(tgt_ids) + 1  # include the final eos as part of the response
    mask = torch.zeros(seq_len, dtype=torch.bool)
    # mask a random fraction of response tokens
    r = random.uniform(0.05, 0.95)
    n_mask = max(1, int(r * n_resp))
    pos = torch.randperm(n_resp)[:n_mask] + len(prompt_ids)
    mask[pos] = True
    return torch.tensor(ids, dtype=torch.long), mask


def run(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer, compute = load_model_and_tokenizer(args)
    model = build_lora(model, args, compute)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "steps": args.steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "load_in_4bit": args.load_in_4bit,
        "created_at": time.time(),
    }
    (out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = compute in (torch.float16, torch.bfloat16) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and compute == torch.float16)

    it = make_example_iterator(args, tokenizer)

    step = 0
    opt_step = 0
    tot_loss = 0.0
    t0 = time.time()
    last_save = 0
    last_eval = 0

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
        prog = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
        min_lr = args.learning_rate * args.min_lr_ratio
        return min_lr + 0.5 * (args.learning_rate - min_lr) * (1 + math.cos(math.pi * prog))

    def save_adapter():
        model.save_pretrained(str(out_dir / "adapter"))
        tokenizer.save_pretrained(str(out_dir / "adapter"))
        print(f"[sft] adapter saved to {out_dir / 'adapter'}")

    def eval_sample():
        # LLaDA-MoE does not support HF model.generate(); instead we report a
        # holdout-style masked-loss estimate on fresh examples.
        import torch.nn.functional as F

        try:
            model.eval()
            losses = []
            with torch.no_grad(), torch.autocast("cuda", dtype=compute, enabled=use_amp):
                for _ in range(min(4, args.batch_size)):
                    ex = next(it, None)
                    if ex is None:
                        break
                    enc = encode_example(tokenizer, ex, args.max_seq_len)
                    if enc is None:
                        continue
                    i, m = enc
                    ids = i.unsqueeze(0).to("cuda")
                    mask = m.unsqueeze(0).to("cuda")
                    logits = model(input_ids=ids).logits
                    flat = logits.view(-1, logits.size(-1))[mask.view(-1)]
                    tgt = ids.view(-1)[mask.view(-1)]
                    losses.append(F.cross_entropy(flat, tgt).item())
            if losses:
                print(f"[sft] holdout loss @ step {step}: {sum(losses) / len(losses):.4f}")
            model.train()
        except Exception as e:
            print(f"[sft] (eval skipped: {e})")
            model.train()

    while step < args.steps:
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch_ids, batch_mask = [], []
        while len(batch_ids) < args.batch_size:
            ex = next(it, None)
            if ex is None:
                it = make_example_iterator(args, tokenizer)
                ex = next(it)
            enc = encode_example(tokenizer, ex, args.max_seq_len)
            if enc is None:
                continue
            i, m = enc
            batch_ids.append(i)
            batch_mask.append(m)
        ids = torch.stack(batch_ids).to("cuda")
        mask = torch.stack(batch_mask).to("cuda")

        with torch.autocast("cuda", dtype=compute, enabled=use_amp):
            logits = model(input_ids=ids).logits  # (B, L, V)
            flat_logits = logits.view(-1, logits.size(-1))[mask.view(-1)]
            flat_tgt = ids.view(-1)[mask.view(-1)]
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_tgt)

        scaler.scale(loss).backward()
        tot_loss += loss.item()

        step += 1
        if step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

        if step % args.log_every == 0:
            avg = tot_loss / min(step, args.log_every)
            print(f"[sft] step {step}/{args.steps} | loss {avg:.4f} | lr {lr:.2e} | {int(time.time()-t0)}s")
            tot_loss = 0.0

        if step - last_save >= args.save_every:
            save_adapter()
            last_save = step
        if args.eval_sample_every and step - last_eval >= args.eval_sample_every:
            eval_sample()
            last_eval = step

    save_adapter()

    # optional: merge and push
    if args.push_repo:
        print("[sft] merging LoRA and pushing...")
        merged = model.merge_and_unload()
        merged = merged.to(torch.float16 if compute == torch.float16 else torch.bfloat16)
        merged.save_pretrained(str(out_dir / "merged"), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir / "merged"))
        from huggingface_hub import HfApi

        api = HfApi(token=args.hf_token or os.environ.get("HF_TOKEN"))
        api.create_repo(args.push_repo, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir / "merged"), repo_id=args.push_repo)
        print(f"[sft] pushed merged model to {args.push_repo}")


if __name__ == "__main__":
    run(parse_args())
