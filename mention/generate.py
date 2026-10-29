"""Parallel iterative unmasking (diffusion decoding).

Starts from a fully masked continuation and over `steps` rounds reveals more
tokens in parallel: sample candidates for masked positions, keep the most
confident ones, remask the rest. Prompt tokens stay fixed.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from .model import MaskedDiffusionTransformer
from .tokenizer import Tokenizer, mask_id


@torch.no_grad()
def generate(
    model: MaskedDiffusionTransformer,
    tok: Tokenizer,
    prompt_ids: List[int],
    gen_len: int,
    steps: int = 8,
    tau: float = 0.7,
    top_k: int = 128,
    keep_frac: float = 0.5,
    device: str = "cpu",
) -> List[int]:
    model.eval()
    model.to(device)
    mask_tok = mask_id(tok)
    L = len(prompt_ids) + gen_len
    seq = torch.full((1, L), mask_tok, dtype=torch.long, device=device)
    if prompt_ids:
        seq[0, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    unmasked = seq != mask_tok  # [1, L]

    for _ in range(steps):
        logits = model(seq)  # [1, L, V]
        probs = F.softmax(logits / tau, dim=-1)
        # sample candidates with top-k filtering
        if top_k > 0:
            top_probs, top_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
            # renormalize over the top-k set
            top_probs = top_probs / top_probs.sum(-1, keepdim=True)
            flat = torch.multinomial(top_probs.view(-1, top_probs.size(-1)), 1).view(1, L, 1)
            cand = torch.gather(top_idx, -1, flat).squeeze(-1)  # [1, L]
        else:
            cand = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, L)

        conf = probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)  # [1, L]
        masked_pos = ~unmasked
        if not masked_pos.any():
            break

        # among still-masked positions, keep the `keep_frac` most confident
        conf_masked = conf.masked_fill(unmasked, -1.0)
        n_keep = max(1, int(masked_pos.sum().item() * keep_frac))
        n_keep = min(n_keep, int(masked_pos.sum().item()))
        keep_idx = conf_masked.view(-1).topk(n_keep).indices  # flat positions
        q = keep_idx // L
        r = keep_idx % L
        seq[q, r] = cand[q, r]
        unmasked[q, r] = True

    return seq[0].tolist()


def decode(tok: Tokenizer, ids: List[int]) -> str:
    return tok.decode(ids)  # type: ignore[union-attr]
