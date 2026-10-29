"""Streaming coding corpus: open code datasets (HF) + a local private folder.

Used when config `data.mode == "code"` (the "Mention 1" coding run).
Tokenizing happens on the fly with the fast BPE tokenizer. Rows are packed
to exactly seq_len tokens; partial remains are carried over to the next row.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import IterableDataset

from .tokenizer import Tokenizer

#: file extensions (lowercase) treated as source code from data/private/
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts",
    ".lua", ".r", ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".dockerfile", ".makefile",
}
#: open-source code datasets that stream without extra auth (adapt if gated).
DEFAULT_CODE_SOURCES = [
    {"dataset": "codeparrot/github-code", "config": "python", "field": "code", "prob": 0.45},
    {"dataset": "codeparrot/github-code", "config": "javascript", "field": "code", "prob": 0.25},
    {"dataset": "codeparrot/github-code", "config": "java", "field": "code", "prob": 0.15},
]
#: cap on a single file/document fed to the tokenizer (bytes/chars).
MAX_DOC_CHARS = 512_000


class StreamingCodeCorpus(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_len: int,
        sources: Optional[List[Dict[str, Any]]] = None,
        private_dir: str = "data/private",
        private_prob: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sources = sources if sources is not None else DEFAULT_CODE_SOURCES
        self.private_dir = Path(private_dir)
        self.private_prob = private_prob

    # ---- source streams --------------------------------------------------
    def _make_source_stream(self, src: Dict[str, Any]) -> Iterator[str]:
        from datasets import load_dataset

        kwargs: Dict[str, Any] = dict(split="train", streaming=True)
        if src.get("config"):
            kwargs["name"] = src["config"]
        ds = load_dataset(src.get("dataset"), **kwargs)  # type: ignore[call-overload]
        field = src.get("field", "text")
        skip = random.randint(0, 50000)
        import itertools

        for ex in itertools.islice(ds, skip, None):
            text = ex.get(field) or ex.get("text") or ex.get("content")
            if isinstance(text, str) and text.strip():
                yield text[:MAX_DOC_CHARS]

    def _make_private_stream(self) -> Iterator[str]:
        """Yield text contents of code files found under data/private/."""
        if not self.private_dir.exists():
            return
        for p in sorted(self.private_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in CODE_EXTENSIONS:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if text.strip():
                yield text[:MAX_DOC_CHARS]

    # ---- iter ------------------------------------------------------------
    def __iter__(self) -> Iterator[List[int]]:
        sources: List[Iterator[str]] = [self._make_source_stream(s) for s in self.sources]
        has_private = self.private_dir.exists()
        if has_private:
            sources.append(self._make_private_stream())
        probs: List[float] = [s.get("prob", 1.0) for s in self.sources]
        if has_private:
            probs.append(self.private_prob)
        total = sum(probs)
        probs = [p / total for p in probs]

        def _next_text() -> str:
            nonlocal sources
            while True:
                idx = random.choices(range(len(sources)), weights=probs, k=1)[0]
                try:
                    return next(sources[idx])
                except StopIteration:
                    if idx < len(self.sources):
                        sources[idx] = self._make_source_stream(self.sources[idx])
                    else:
                        sources[idx] = self._make_private_stream()

        # document buffer: sample randomly, refresh 5% each draw
        docbuf: List[str] = [_next_text() for _ in range(1024)]
        buf: List[int] = []
        while True:
            idx = random.randrange(len(docbuf))
            text = docbuf[idx]
            if random.random() < 0.05:
                docbuf[idx] = _next_text()
            ids = self.tokenizer.encode(text).ids  # type: ignore[union-attr]
            buf.extend(ids)
            while len(buf) >= self.seq_len:
                row = buf[: self.seq_len]
                del buf[: self.seq_len]
                yield row


def iter_raw_code_texts(
    sources: Optional[List[Dict[str, Any]]] = None,
    private_dir: str = "data/private",
    limit: Optional[int] = None,
) -> Iterator[str]:
    """Stream raw code text without a tokenizer (used to train a code tokenizer)."""
    cfg = StreamingCodeCorpus.__new__(StreamingCodeCorpus)
    cfg.sources = sources if sources is not None else DEFAULT_CODE_SOURCES
    cfg.private_dir = Path(private_dir)
    streams: List[Iterator[str]] = [cfg._make_source_stream(s) for s in cfg.sources]
    if cfg.private_dir.exists():
        streams.append(cfg._make_private_stream())
    for it in streams:
        for text in it:
            yield text
            if limit is not None:
                limit -= 1
                if limit <= 0:
                    return
