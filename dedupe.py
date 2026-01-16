import asyncio
import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Iterable, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class EmbedEntry:
    ts: float
    tickers: Tuple[str, ...]
    vec: np.ndarray


class Dedupe:
    """Embedding-only dedupe using OpenAI text-embedding-3-small."""

    def __init__(
        self,
        api_key: str,
        ttl: float = 90.0,
        sim_threshold: float = 0.85,
    ) -> None:
        self.api_key = api_key
        self.ttl = ttl
        self.sim_threshold = sim_threshold

        self._embeds: deque[EmbedEntry] = deque()
        self._hashes: set[str] = set()
        self._hash_store = Path("dedupe_hashes.jsonl")
        self._load_hashes()
        self._lock = asyncio.Lock()
        self._client = AsyncOpenAI(api_key=self.api_key)

    def _prune(self, now: float) -> None:
        while self._embeds and now - self._embeds[0].ts > self.ttl:
            self._embeds.popleft()

    def _load_hashes(self) -> None:
        if not self._hash_store.exists():
            return
        try:
            with self._hash_store.open("r", encoding="utf-8") as f:
                for line in f:
                    h = line.strip()
                    if h:
                        self._hashes.add(h)
        except Exception as exc:
            logger.warning("Failed to load dedupe hashes: %s", exc)

    def _persist_hash(self, h: str) -> None:
        try:
            with self._hash_store.open("a", encoding="utf-8") as f:
                f.write(h + "\n")
        except Exception as exc:
            logger.warning("Failed to persist dedupe hash: %s", exc)

    async def _embed(self, text: str) -> np.ndarray:
        resp = await self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        # Normalize to unit vector for cosine via dot product
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    async def accept(self, headline: str, tickers: Iterable[str]) -> tuple[bool, str]:
        """
        Returns (accepted, reason).
        reason is "pass" if accepted, otherwise stage label.
        """
        tks = tuple(sorted({t.strip().upper() for t in tickers if t}))
        now = monotonic()

        try:
            vec = await self._embed(headline)
        except Exception as exc:
            logger.warning("Embed failed, passing through: %s", exc)
            return True, "embed_error"

        max_sim = 0.0
        async with self._lock:
            now = monotonic()
            self._prune(now)

            # Stage 0: exact hash
            h = hashlib.sha256(headline.encode("utf-8")).hexdigest()
            if h in self._hashes:
                logger.info("Dedup hash hit headline=%s", headline)
                return False, "hash"
            self._hashes.add(h)
            self._persist_hash(h)

            for entry in self._embeds:
                sim = float(np.dot(entry.vec, vec))
                if sim > max_sim:
                    max_sim = sim
                if self.sim_threshold <= sim:
                    logger.info(
                        "Dedup embed hit sim=%.3f tickers=%s headline=%s",
                        sim,
                        ",".join(tks) or "UNK",
                        headline,
                    )
                    return False, "embed"

            self._embeds.append(EmbedEntry(ts=now, tickers=tks or ("UNK",), vec=vec))
            return True, f"pass(sim_max={max_sim:.3f})"
