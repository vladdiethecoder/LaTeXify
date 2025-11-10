#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

LIX_REAL_CLASSES = {"textbook", "novella", "newspaper", "contract"}

# Map our planner’s symbolic names to actual .cls
_CANON = {
    "lix": "textbook",
    "lix_article": "textbook",
    "lix_textbook": "textbook",
    # pass-throughs if planner already emits a real LiX class
    "textbook": "textbook",
    "novella": "novella",
    "newspaper": "newspaper",
    "contract": "contract",
    # Standards we already supported
    "article": "article",
    "scrartcl": "scrartcl",
}

@dataclass(frozen=True)
class DocClassDecision:
    original: str
    normalized: str
    requires_lix: bool

def normalize_docclass(name: str) -> DocClassDecision:
    original = (name or "").strip()
    key = original.lower()
    normalized = _CANON.get(key, key or "scrartcl")
    requires_lix = normalized in LIX_REAL_CLASSES
    return DocClassDecision(original=original or "scrartcl", normalized=normalized, requires_lix=requires_lix)
