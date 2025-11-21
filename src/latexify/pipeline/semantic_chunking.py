"""Semantic chunking utilities for PlannerAgent-style breakpoints."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

def _have_transformer_checkpoint(name: str) -> bool:
    cache_home = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")
    repo_slug = name.replace("/", "_")
    target = Path(cache_home) / repo_slug
    return target.exists() and any(target.rglob("config.json"))

LOGGER = logging.getLogger(__name__)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
DEFAULT_BACKEND = os.environ.get("LATEXIFY_SEMANTIC_CHUNKER_BACKEND", "auto")
FORCE_CHUNKER_DOWNLOAD = os.environ.get("LATEXIFY_SEMANTIC_CHUNKER_FORCE_DOWNLOAD", "0") == "1"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ALLOW_HASH_FALLBACK = _env_flag("LATEXIFY_SEMANTIC_CHUNKER_ALLOW_FALLBACK", False)


def _count_sentences(text: str) -> int:
    if not text.strip():
        return 0
    parts = SENTENCE_RE.split(text.strip())
    return max(1, len([p for p in parts if p.strip()]))


class _BaseEncoder:
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError


class _HashingEncoder(_BaseEncoder):
    """Fallback encoder that approximates semantics via hashed bag-of-words."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        vectors: List[torch.Tensor] = []
        for text in texts:
            vec = torch.zeros(self.dim, dtype=torch.float32)
            for token in WORD_RE.findall(text.lower()):
                idx = abs(hash(token)) % self.dim
                vec[idx] += 1.0
            if vec.norm(p=2) > 0:
                vec = F.normalize(vec, p=2, dim=0)
            vectors.append(vec)
        return torch.stack(vectors, dim=0) if vectors else torch.zeros((0, self.dim))


class _TransformerEncoder(_BaseEncoder):  # pragma: no cover - heavy dependency
    """Sentence embedding encoder backed by a local HuggingFace transformer."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 384,
        device: str | None = None,
        allow_download: bool = False,
    ) -> None:
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError("transformers is unavailable")
        local_only = not allow_download
        kwargs = {"local_files_only": local_only}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            self.model = AutoModel.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            if allow_download or FORCE_CHUNKER_DOWNLOAD:
                raise
            if _have_transformer_checkpoint(model_name):
                raise
            LOGGER.warning(
                "Semantic chunker transformer unavailable (%s); falling back to hashing encoder.",
                exc,
            )
            raise RuntimeError("transformer-missing")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.model.config.hidden_size))
        with torch.no_grad():
            batch = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            token_embeddings = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            sentence_embeddings = summed / counts
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu()


class SemanticChunkerError(RuntimeError):
    """Raised when the semantic chunker cannot load its transformer encoder."""


@dataclass
class SemanticChunker:
    """Detects semantic breakpoints using cosine distance between embeddings."""

    distance_threshold: float = 0.42
    min_sentences_per_chunk: int = 2
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_backend: str = field(default_factory=lambda: DEFAULT_BACKEND)  # auto | hf | hash
    allow_model_download: bool = True
    allow_hash_fallback: bool = field(default_factory=lambda: ALLOW_HASH_FALLBACK)

    def __post_init__(self) -> None:
        self.encoder = self._build_encoder()

    def _build_encoder(self) -> _BaseEncoder:
        backend = self.encoder_backend.lower()
        if backend == "hash":
            LOGGER.info("SemanticChunker using hashing encoder backend")
            return _HashingEncoder()
        if backend in {"hf", "auto"}:
            allow_download = self.allow_model_download or bool(
                os.environ.get("LATEXIFY_SEMANTIC_CHUNKER_ALLOW_DOWNLOAD")
            )
            try:
                LOGGER.info(
                    "SemanticChunker loading transformer %s (download=%s)",
                    self.encoder_name,
                    allow_download,
                )
                return _TransformerEncoder(
                    self.encoder_name,
                    allow_download=allow_download,
                )
            except RuntimeError as exc:
                message = (
                    f"SemanticChunker failed to load '{self.encoder_name}' from cache. "
                    "Install it via `huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir <path>` "
                    "or allow downloads by setting LATEXIFY_SEMANTIC_CHUNKER_ALLOW_DOWNLOAD=1."
                )
                if backend == "hf" or not self.allow_hash_fallback:
                    raise SemanticChunkerError(message) from exc
                LOGGER.warning("%s; falling back to hashing encoder backend", message)
        if not self.allow_hash_fallback and backend == "hash":
            raise SemanticChunkerError(
                "Hashing backend explicitly requested but allow_hash_fallback=False; "
                "either enable LATEXIFY_SEMANTIC_CHUNKER_ALLOW_FALLBACK=1 or use encoder_backend='hf'."
            )
        if not self.allow_hash_fallback:
            raise SemanticChunkerError(
                "SemanticChunker hashing fallback is disabled. "
                "Set LATEXIFY_SEMANTIC_CHUNKER_ALLOW_FALLBACK=1 to permit hash mode."
            )
        LOGGER.info("SemanticChunker falling back to hashing encoder backend")
        return _HashingEncoder()

    def sentence_count(self, text: str) -> int:
        return _count_sentences(text)

    def embed(self, text: str) -> torch.Tensor | None:
        stripped = text.strip()
        if not stripped:
            return None
        vec = self.encoder.encode([stripped])
        return vec[0] if len(vec) else None

    def should_break(
        self,
        prev_embedding: torch.Tensor | None,
        current_embedding: torch.Tensor | None,
        buffer_sentence_count: int,
    ) -> bool:
        if prev_embedding is None or current_embedding is None:
            return False
        if buffer_sentence_count < self.min_sentences_per_chunk:
            return False
        similarity = torch.dot(prev_embedding, current_embedding).clamp(-1.0, 1.0).item()
        distance = 1.0 - similarity
        return distance >= self.distance_threshold


__all__ = ["SemanticChunker", "SemanticChunkerError"]
