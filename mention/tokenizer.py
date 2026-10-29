"""Fast BPE tokenizer (tokenizers lib) for RU+EN text.

Special tokens:
  <|pad|>  -> id 0 (also used as the diffusion MASK token)
  <|mask|> -> id 1 (explicit mask token, reserved for future use)
  <|unk|>  -> id 2
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

PAD_TOKEN = "<|pad|>"
MASK_TOKEN = "<|mask|>"
UNK_TOKEN = "<|unk|>"


def train_bpe(
    files: Iterable[str | Path],
    vocab_size: int,
    out_path: str | Path,
) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN, MASK_TOKEN, UNK_TOKEN],
        min_frequency=2,
        show_progress=True,
    )
    tok.train([str(f) for f in files], trainer)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def tokenize_texts(tok: Tokenizer, texts: Iterable[str], max_len: int) -> List[List[int]]:
    """Encode a batch/list of texts into token id lists (no BOS/EOS)."""
    return [
        tok.encode(t).ids[:max_len]  # type: ignore[union-attr]
        for t in texts
    ]


def mask_id(tok: Tokenizer) -> int:
    return int(tok.token_to_id(MASK_TOKEN))  # type: ignore[arg-type]


def pad_id(tok: Tokenizer) -> int:
    return int(tok.token_to_id(PAD_TOKEN))  # type: ignore[arg-type]
