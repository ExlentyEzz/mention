#!/usr/bin/env python3
"""Train the BPE tokenizer on a sample of the EN+RU corpus.

Pulls up to --samples-per-source texts from fineweb-edu (EN) and mc4/ru (RU)
in streaming mode, trains a ByteLevel BPE with vocab_size, and saves:
  <out>/tokenizer.json        (the tokenizer itself)
  <out>/vocab_info.json       (size + special token ids)

Usage:
  python scripts/train_tokenizer.py --vocab-size 49152 --out artifacts/tokenizer
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mention.tokenizer import (  # noqa: E402
    MASK_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    train_bpe,
)


def stream_texts(dataset: str, config: str | None, limit: int):
    from datasets import load_dataset

    kwargs = dict(split="train", streaming=True, trust_remote_code=True)
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset, **kwargs)  # type: ignore[call-overload]
    for ex in itertools.islice(ds, limit):
        text = ex.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=49152)
    p.add_argument("--out", default="artifacts/tokenizer")
    p.add_argument("--samples-per-source", type=int, default=200000)
    p.add_argument("--en-dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--ru-dataset", default="mc4")
    p.add_argument("--ru-config", default="ru")
    p.add_argument("--min-frequency", type=int, default=2)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tokenizer] sampling EN from {args.en_dataset} ...")
    en = stream_texts(args.en_dataset, None, args.samples_per_source)
    print(f"[tokenizer] sampling RU from {args.ru_dataset}/{args.ru_config} ...")
    ru = stream_texts(args.ru_dataset, args.ru_config, args.samples_per_source)
    texts = itertools.chain(en, ru)

    import tokenizers
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizers import decoders, normalizers

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[PAD_TOKEN, MASK_TOKEN, UNK_TOKEN],
        min_frequency=args.min_frequency,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer)

    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path))

    info = {
        "vocab_size": int(tok.get_vocab_size()),
        "pad_id": tok.token_to_id(PAD_TOKEN),
        "mask_id": tok.token_to_id(MASK_TOKEN),
        "unk_id": tok.token_to_id(UNK_TOKEN),
        "en_dataset": args.en_dataset,
        "ru_dataset": f"{args.ru_dataset}/{args.ru_config}",
        "samples_per_source": args.samples_per_source,
        "min_frequency": args.min_frequency,
    }
    (out_dir / "vocab_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[tokenizer] done -> {tok_path}")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
