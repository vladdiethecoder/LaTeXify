"""Infer academic domain to drive semantic enrichment and packages."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from ..core import common

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "mathematics": ["theorem", "lemma", "proof", "corollary", "equation", "matrix", "vector"],
    "computer_science": ["algorithm", "runtime", "complexity", "graph", "procedure", "pseudocode"],
    "physics": ["quantum", "energy", "electron", "momentum", "field", "particle"],
    "engineering": ["signal", "circuit", "controller", "thermal", "mechanical", "stress"],
}

DOMAIN_PACKAGES: Dict[str, List[Dict[str, str | None]]] = {
    "mathematics": [{"package": "amsthm"}, {"package": "mathtools"}],
    "computer_science": [{"package": "algorithm2e"}, {"package": "algpseudocode"}],
    "physics": [{"package": "siunitx"}, {"package": "physics"}],
    "engineering": [{"package": "circuitikz"}, {"package": "siunitx"}],
}


@dataclass
class DomainProfile:
    domain: str
    confidence: float
    scores: Dict[str, float]
    features: Dict[str, float]
    recommended_packages: List[Dict[str, str | None]]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["confidence"] = round(self.confidence, 3)
        return payload


class DomainDetector:
    """Simple keyword + structure heuristic for domain inference."""

    def analyze(self, chunks_path: Path, plan_path: Path | None = None) -> DomainProfile:
        chunks = list(common.load_chunks(chunks_path))
        features = self._extract_features(chunks)
        scores = self._score_domains(chunks, features)
        domain, confidence = self._select_domain(scores)
        packages = DOMAIN_PACKAGES.get(domain, [])
        return DomainProfile(
            domain=domain,
            confidence=confidence,
            scores=scores,
            features=features,
            recommended_packages=packages,
        )

    def _extract_features(self, chunks: List[common.Chunk]) -> Dict[str, float]:
        total = max(1, len(chunks))
        math_regions = 0
        algorithm_hints = 0
        citation_hints = 0
        physics_symbols = 0
        for chunk in chunks:
            text = chunk.text.lower()
            metadata = chunk.metadata or {}
            region = metadata.get("region_type", "")
            math_role = metadata.get("math_role", "")
            if region in {"formula", "table"} or "equation" in math_role:
                math_regions += 1
            if any(token in text for token in ("input:", "output:", "procedure", "step")):
                algorithm_hints += 1
            if "\\cite" in chunk.text or "et al." in chunk.text.lower():
                citation_hints += 1
            if any(symbol in text for symbol in ("\\omega", "\\phi", "joule", "newton")):
                physics_symbols += 1
        return {
            "math_density": math_regions / total,
            "algorithm_density": algorithm_hints / total,
            "citation_density": citation_hints / total,
            "physics_density": physics_symbols / total,
        }

    def _score_domains(self, chunks: List[common.Chunk], features: Dict[str, float]) -> Dict[str, float]:
        corpus = " ".join(chunk.text.lower() for chunk in chunks)
        scores: Dict[str, float] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            keyword_hits = sum(corpus.count(keyword) for keyword in keywords)
            normalized = keyword_hits / max(1, len(corpus) // 500)
            if domain == "mathematics":
                normalized += features["math_density"] * 1.5
            elif domain == "computer_science":
                normalized += features["algorithm_density"] * 1.3
            elif domain == "physics":
                normalized += features["physics_density"] * 1.2
            elif domain == "engineering":
                normalized += (features["citation_density"] + features["math_density"]) * 0.5
            scores[domain] = round(normalized, 4)
        return scores

    def _select_domain(self, scores: Dict[str, float]) -> tuple[str, float]:
        if not scores:
            return "general", 0.0
        domain = max(scores, key=lambda key: scores[key])
        total = sum(value for value in scores.values() if value > 0) or 1.0
        confidence = scores[domain] / total if total else 0.0
        return domain, confidence


__all__ = ["DomainDetector", "DomainProfile"]
