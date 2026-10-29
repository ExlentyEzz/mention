"""Streaming pretraining corpus: EN (FineWeb-Edu) + RU (mC4) mixed.

Tokenizing happens on the fly with the fast BPE tokenizer. Rows are packed
to exactly seq_len tokens; partial remains are carried over to the next row.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .tokenizer import Tokenizer


class StreamingLMCorpus(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_len: int,
        en_dataset: str = "HuggingFaceFW/fineweb-edu",
        ru_dataset: str = "HuggingFaceFW/fineweb-2",
        ru_config: str = "rus_Cyrl",
        en_prob: float = 0.55,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.en_dataset = en_dataset
        self.ru_dataset = ru_dataset
        self.ru_config = ru_config
        self.en_prob = en_prob

    def _make_stream(self, dataset: str, config: Optional[str]) -> Iterator[str]:
        from datasets import load_dataset

        kwargs = dict(split="train", streaming=True)
        if config:
            kwargs["name"] = config
        ds = load_dataset(dataset, **kwargs)  # type: ignore[call-overload]
        for ex in ds:
            text = ex.get("text")
            if isinstance(text, str) and text.strip():
                yield text

    def __iter__(self) -> Iterator[List[int]]:
        en_it = self._make_stream(self.en_dataset, None)
        ru_it = self._make_stream(self.ru_dataset, self.ru_config)
        buf: List[int] = []
        while True:
            if random.random() < self.en_prob:
                try:
                    text = next(en_it)
                except StopIteration:
                    en_it = self._make_stream(self.en_dataset, None)
                    text = next(en_it)
            else:
                try:
                    text = next(ru_it)
                except StopIteration:
                    ru_it = self._make_stream(self.ru_dataset, self.ru_config)
                    text = next(ru_it)
            ids = self.tokenizer.encode(text).ids  # type: ignore[union-attr]
            buf.extend(ids)
            while len(buf) >= self.seq_len:
                row = buf[: self.seq_len]
                del buf[: self.seq_len]
                yield row


class TinySmokeCorpus(IterableDataset):
    """Offline toy corpus for smoke tests (no internet/datasets needed)."""

    LINES = [
        "Я люблю учить нейронные сети с нуля, это очень интересно.",
        "The quick brown fox jumps over the lazy dog near the river.",
        "Маскированная диффузия позволяет генерировать токены параллельно.",
        "Diffusion language models generate many tokens at once in parallel.",
        "Сегодня мы обучаем маленькую модель на двух видеокартах Tesla T4.",
        "Kaggle gives free GPU hours every single week for research projects.",
        "Hugging Face Hub stores our shared checkpoints between sessions.",
        "Checkpoint resume lets us continue training from the last saved step.",
    ]

    def __init__(self, tokenizer: Tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[List[int]]:
        buf: List[int] = []
        while True:
            text = random.choice(self.LINES)
            buf.extend(self.tokenizer.encode(text).ids)  # type: ignore[union-attr]
            while len(buf) >= self.seq_len:
                row = buf[: self.seq_len]
                del buf[: self.seq_len]
                yield row


def iter_raw_texts(
    dataset: str,
    config: Optional[str],
    limit: Optional[int] = None,
) -> Iterator[str]:
    """Stream raw 'text' fields without a tokenizer (used by tokenizer training)."""
    from datasets import load_dataset

    kwargs = dict(split="train", streaming=True)
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset, **kwargs)  # type: ignore[call-overload]
    import itertools

    for ex in itertools.islice(ds, limit):
        text = ex.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def make_dataloader(
    corpus: IterableDataset,
    micro_batch: int,
    num_workers: int = 0,
) -> DataLoader:
    def collate(batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(batch, dtype=torch.long)}

    return DataLoader(
        corpus,
        batch_size=micro_batch,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )
