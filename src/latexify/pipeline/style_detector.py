"Automatic document-style detection and template selection helpers."
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

try:  # pragma: no cover - optional dependency for CLI evaluation
    from pypdf import PdfReader
except Exception:  # pragma: no cover - dependency may be missing on CI
    PdfReader = None  # type: ignore

from ..core import common
from ..ml.style_classifier import StyleClassifier  # Integration of new ML module

LOGGER = logging.getLogger(__name__)
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference_tex"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "style_classifier"

CITATION_BRACKET_RE = re.compile(r"\\[[0-9]{1,3}(?:\s*,\s*[0-9]{1,3})*\\]")
CITATION_PAREN_RE = re.compile(r"\([A-Z][A-Za-z-]+\s*,\s*\d{4}\)")
MATH_TOKEN_RE = re.compile(r"\\(begin{equation|frac|sum|int|alpha|beta|gamma|nabla|sqrt)")
SECTION_KEYWORD_RE = re.compile(r"(?i)\b(section|chapter|part)\b")
REFERENCE_RE = re.compile(r"(?i)\b(references|bibliography)\b")

DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "math": ("theorem", "lemma", "proof", "eigenvalue", "manifold"),
    "cs": ("algorithm", "complexity", "neural", "network", "bytecode", "computation"),
    "physics": ("quantum", "particle", "energy", "relativity", "plasma"),
    "bio": ("cell", "protein", "enzyme", "genome", "biomarker"),
    "finance": ("portfolio", "equity", "derivative", "valuation", "yield"),
}

DOC_TYPE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "thesis": ("thesis", "dissertation", "advisor", "committee"),
    "presentation": ("slide", "agenda", "speaker", "overview"),
    "book": ("chapter", "preface", "appendix"),
}

DOC_TYPE_CLASS_MAP: Dict[str, Tuple[str, str]] = {
    "article": ("article", "11pt"),
    "report": ("report", "11pt"),
    "book": ("memoir", "11pt"), # Updated to memoir per blueprint
    "thesis": ("report", "12pt"),
    "presentation": ("beamer", "11pt"),
    "neurips": ("neurips", "10pt"), # Placeholder mapping, template handles actual class
    "iclr": ("iclr", "10pt"),
    "textbook": ("memoir", "10pt"),
}

DOC_TYPE_PACKAGES: Dict[str, List[Dict[str, str | None]]] = {
    "article": [{"package": "amsmath"}, {"package": "amssymb"}],
    "report": [{"package": "titlesec"}],
    "book": [{"package": "tocloft"}],
    "thesis": [{"package": "setspace", "options": "onehalfspacing"}],
    "presentation": [{"package": "xcolor"}],
    "neurips": [], # Handled by template
    "iclr": [],
    "textbook": [{"package": "microtype"}]
}

STYLE_FAMILY_PACKAGES: Dict[str, List[Dict[str, str | None]]] = {
    "ieee": [{"package": "cite"}, {"package": "balance"}],
    "acm": [{"package": "ragged2e"}, {"package": "microtype"}],
    "springer": [{"package": "mathptmx"}],
    "generic": [],
}

DOMAIN_PACKAGES: Dict[str, List[Dict[str, str | None]]] = {
    "math": [{"package": "mathtools"}],
    "cs": [{"package": "algorithm2e", "options": "ruled,linesnumbered"}],
    "physics": [{"package": "siunitx"}],
    "bio": [{"package": "mhchem"}],
    "finance": [{"package": "pgfplots"}],
}


def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    norm_a = math.sqrt(sum(value * value for value in vec_a)) or 1.0
    norm_b = math.sqrt(sum(value * value for value in vec_b)) or 1.0
    return sum(a * b for a, b in zip(vec_a, vec_b)) / (norm_a * norm_b)


@dataclass
class StyleFeatures:
    total_blocks: int = 1
    section_blocks: int = 0
    figure_blocks: int = 0
    table_blocks: int = 0
    equation_blocks: int = 0
    text_tokens: int = 0
    math_tokens: int = 0
    bracket_citations: int = 0
    paren_citations: int = 0
    reference_markers: int = 0
    page_estimate: int = 1

    def vector(self) -> List[float]:
        blocks = max(1, self.total_blocks)
        tokens = max(1, self.text_tokens)
        return [
            self.section_blocks / blocks,
            self.figure_blocks / blocks,
            self.table_blocks / blocks,
            self.equation_blocks / blocks,
            self.math_tokens / tokens,
            self.bracket_ratio(),
            self.paren_ratio(),
            self.reference_markers / blocks,
            min(1.0, self.page_estimate / 120.0),
        ]

    def bracket_ratio(self) -> float:
        total = self.bracket_citations + self.paren_citations or 1
        return self.bracket_citations / total

    def paren_ratio(self) -> float:
        total = self.bracket_citations + self.paren_citations or 1
        return self.paren_citations / total

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_blocks": self.total_blocks,
            "section_blocks": self.section_blocks,
            "figure_blocks": self.figure_blocks,
            "table_blocks": self.table_blocks,
            "equation_blocks": self.equation_blocks,
            "text_tokens": self.text_tokens,
            "math_tokens": self.math_tokens,
            "bracket_citations": self.bracket_citations,
            "paren_citations": self.paren_citations,
            "reference_markers": self.reference_markers,
            "page_estimate": self.page_estimate,
        }


@dataclass
class DocumentStyle:
    doc_type: str
    style_family: str
    domain: str
    reference_domain: str
    document_class: str
    class_options: str
    packages: List[Dict[str, str | None]]
    template_hint: str | None
    similarity: float
    features: StyleFeatures
    similarity_breakdown: Dict[str, float] = field(default_factory=dict)
    ml_confidence: float = 0.0  # Added confidence score

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_type": self.doc_type,
            "style_family": self.style_family,
            "domain": self.domain,
            "reference_domain": self.reference_domain,
            "document_class": self.document_class,
            "class_options": self.class_options,
            "packages": self.packages,
            "template_hint": self.template_hint,
            "similarity": round(self.similarity, 3),
            "ml_confidence": round(self.ml_confidence, 3),
            "features": self.features.to_dict(),
            "similarity_breakdown": self.similarity_breakdown,
        }


class StyleDetector:
    """
    Detect document style using a hybrid approach:
    1. ML-based classification (StyleClassifier) for high-level types (NeurIPS, ICLR, Textbook).
    2. Heuristic fallbacks for domains, features, and package selection.
    """

    def __init__(self, reference_root: Path | None = None, model_dir: Path | None = None) -> None:
        self.reference_root = reference_root or REFERENCE_ROOT
        self.reference_profiles = self._load_reference_profiles()
        
        self.classifier = StyleClassifier(model_dir=model_dir or MODEL_DIR)
        self.classifier_loaded = self.classifier.load()

    # --------------------------- public entry points ---------------------------
    def detect_from_plan(
        self,
        plan: Sequence[common.PlanBlock],
        chunk_map: Dict[str, common.Chunk] | None = None,
    ) -> DocumentStyle:
        features, samples = self._features_from_plan(plan, chunk_map or {})
        return self._build_profile(features, samples)

    def detect_style(self, source: str | Path, *, max_pages: int = 12) -> Dict[str, object]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Style detection source {path} missing")
        if path.suffix.lower() != ".pdf":
            raise ValueError("StyleDetector.detect_style expects a PDF path for CLI usage")
        if PdfReader is None:
            raise RuntimeError("pypdf is not available; install it to enable PDF style detection")
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        texts: List[str] = []
        for idx in range(min(max_pages, page_count)):
            try:
                texts.append(reader.pages[idx].extract_text() or "")
            except Exception:
                continue
        text_blob = "\n".join(texts)
        features = self._features_from_text(text_blob, page_estimate=max(1, page_count))
        profile = self._build_profile(features, [text_blob])
        return profile.to_dict()

    # --------------------------- feature extraction ---------------------------
    def _features_from_plan(
        self,
        plan: Sequence[common.PlanBlock],
        chunk_map: Dict[str, common.Chunk],
    ) -> Tuple[StyleFeatures, List[str]]:
        page_estimate = max((chunk.page for chunk in chunk_map.values()), default=len(plan) // 4 or 1)
        features = StyleFeatures(total_blocks=max(1, len(plan)), page_estimate=max(1, page_estimate))
        samples: List[str] = []
        for block in plan:
            if block.block_type == "section":
                features.section_blocks += 1
            if block.block_type == "figure":
                features.figure_blocks += 1
            if block.block_type == "table":
                features.table_blocks += 1
            if block.block_type == "equation":
                features.equation_blocks += 1
            chunk = chunk_map.get(block.chunk_id)
            if not chunk:
                continue
            samples.append(chunk.text)
            tokens = len(chunk.text.split())
            features.text_tokens += tokens
            features.math_tokens += chunk.text.count("$") + len(MATH_TOKEN_RE.findall(chunk.text))
            features.bracket_citations += len(CITATION_BRACKET_RE.findall(chunk.text))
            features.paren_citations += len(CITATION_PAREN_RE.findall(chunk.text))
            features.reference_markers += len(REFERENCE_RE.findall(chunk.text))
        return features, samples

    def _features_from_text(self, text: str, page_estimate: int = 1) -> StyleFeatures:
        sections = len(SECTION_KEYWORD_RE.findall(text))
        figures = text.lower().count("figure")
        tables = text.lower().count("table")
        equations = text.count("=") // max(1, text.count(".\n"))
        features = StyleFeatures(
            total_blocks=max(1, sections + figures + tables + equations),
            section_blocks=sections,
            figure_blocks=figures // 2,
            table_blocks=tables // 2,
            equation_blocks=equations,
            text_tokens=len(text.split()),
            math_tokens=len(MATH_TOKEN_RE.findall(text)) + text.count("$") * 2,
            bracket_citations=len(CITATION_BRACKET_RE.findall(text)),
            paren_citations=len(CITATION_PAREN_RE.findall(text)),
            reference_markers=len(REFERENCE_RE.findall(text)),
            page_estimate=max(1, page_estimate),
        )
        return features

    # --------------------------- classification ---------------------------
    def _build_profile(self, features: StyleFeatures, text_samples: Sequence[str]) -> DocumentStyle:
        merged_text = "\n".join(text_samples)
        
        # Hybrid Classification Strategy
        ml_doc_type = None
        ml_confidence = 0.0
        
        if self.classifier_loaded:
            # We use the first 512 chars roughly or whatever the model handles (it truncates internally)
            prediction = self.classifier.predict(merged_text[:2000]) 
            if prediction.confidence > 0.7: # Threshold for trusting ML over heuristics
                ml_doc_type = prediction.style_label
                ml_confidence = prediction.confidence
                LOGGER.info(f"ML Style Classifier detected: {ml_doc_type} (conf: {ml_confidence:.2f})")

        if ml_doc_type:
            doc_type = ml_doc_type
        else:
            doc_type = self._classify_doc_type(features, merged_text)
            LOGGER.info(f"Heuristic Style Classifier detected: {doc_type}")

        style_family = self._infer_style_family(features, doc_type)
        domain = self._infer_domain(merged_text) or "default"
        ref_domain, similarity, breakdown = self._match_reference_domain(features)
        document_class, class_options = self._map_doc_class(doc_type, style_family)
        packages = self._merge_packages(
            DOC_TYPE_PACKAGES.get(doc_type, []),
            STYLE_FAMILY_PACKAGES.get(style_family, []),
            DOMAIN_PACKAGES.get(domain, []),
        )
        
        # Map doc_type directly to template if it exists
        template_hint = None
        if doc_type in ["neurips", "iclr", "textbook"]:
            template_hint = f"{doc_type}.tex" # Direct template mapping
        elif ref_domain and ref_domain != "default":
            template_hint = str(self.reference_root / ref_domain)

        return DocumentStyle(
            doc_type=doc_type,
            style_family=style_family,
            domain=domain,
            reference_domain=ref_domain,
            document_class=document_class,
            class_options=class_options,
            packages=packages,
            template_hint=template_hint,
            similarity=similarity,
            features=features,
            similarity_breakdown=breakdown,
            ml_confidence=ml_confidence
        )

    def _classify_doc_type(self, features: StyleFeatures, text: str) -> str:
        lower = text.lower()
        if any(keyword in lower for keyword in DOC_TYPE_KEYWORDS["thesis"]) and features.page_estimate >= 30:
            return "thesis"
        if any(keyword in lower for keyword in DOC_TYPE_KEYWORDS["book"]) and features.page_estimate >= 50:
            return "book"
        if features.page_estimate >= 80:
            return "book"
        if features.figure_blocks / max(1, features.total_blocks) < 0.15 and features.section_blocks >= max(4, int(0.4 * features.total_blocks)):
            return "report"
        if any(keyword in lower for keyword in DOC_TYPE_KEYWORDS["presentation"]):
            return "presentation"
        if features.figure_blocks / max(1, features.total_blocks) > 0.35 and features.page_estimate <= 15:
            return "presentation"
        return "article"

    def _infer_style_family(self, features: StyleFeatures, doc_type: str) -> str:
        if features.bracket_ratio() > 0.55 and features.section_blocks >= 3:
            return "ieee"
        if features.paren_ratio() > 0.45 and doc_type in {"article", "report"}:
            return "acm"
        if doc_type in {"book", "thesis", "textbook"}:
            return "springer"
        return "generic"

    def _infer_domain(self, text: str) -> str | None:
        lower = text.lower()
        best_domain = None
        best_score = 0
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(lower.count(keyword) for keyword in keywords)
            if score > best_score:
                best_domain = domain
                best_score = score
        return best_domain

    def _match_reference_domain(self, features: StyleFeatures) -> Tuple[str, float, Dict[str, float]]:
        if not self.reference_profiles:
            return "default", 0.0, {}
        doc_vec = features.vector()
        breakdown: Dict[str, float] = {}
        best_domain = "default"
        best_score = -1.0
        for domain, ref_vec in self.reference_profiles.items():
            score = _cosine(doc_vec, ref_vec)
            breakdown[domain] = round(score, 3)
            if score > best_score:
                best_domain = domain
                best_score = score
        return best_domain, max(0.0, best_score), breakdown

    def _map_doc_class(self, doc_type: str, style_family: str) -> Tuple[str, str]:
        base_class, options = DOC_TYPE_CLASS_MAP.get(doc_type, ("article", "11pt"))
        if style_family == "ieee":
            options = "10pt,twocolumn"
        elif style_family == "acm":
            options = "manuscript"
        elif style_family == "presentation":
            options = "professionalfonts"
        return base_class, options

    def _merge_packages(self, *groups: Iterable[Dict[str, str | None]]) -> List[Dict[str, str | None]]:
        merged: List[Dict[str, str | None]] = []
        seen: set[str] = set()
        for group in groups:
            for spec in group:
                name = spec.get("package")
                if not name or name in seen:
                    continue
                seen.add(name)
                merged.append({"package": name, "options": spec.get("options")})
        return merged

    # --------------------------- reference profiling ---------------------------
    def _load_reference_profiles(self) -> Dict[str, List[float]]:
        profiles: Dict[str, List[float]] = {}
        root = self.reference_root
        if not root.exists():
            LOGGER.debug("reference_tex root %s missing; style similarity disabled", root)
            return profiles
        for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            vectors: List[List[float]] = []
            for tex_file in domain_dir.glob("*.tex"):
                try:
                    text = tex_file.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                features = self._features_from_text(text, page_estimate=5)
                vectors.append(features.vector())
            if vectors:
                averages = [sum(values) / len(vectors) for values in zip(*vectors)]
                profiles[domain_dir.name] = averages
        if "default" not in profiles:
            profiles["default"] = [0.2, 0.1, 0.05, 0.1, 0.2, 0.5, 0.5, 0.1, 0.2]
        return profiles


__all__ = ["StyleDetector", "DocumentStyle", "StyleFeatures"]