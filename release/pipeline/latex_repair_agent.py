"""Latex repair agent powered by the local Kimi-K2 adapter."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

from ..models.kimi_k2_adapter import LATEX_VALIDATION_GRAMMAR, get_kimi_adapter
from . import kimi_metrics

LOGGER = logging.getLogger(__name__)

UNBALANCED_ENV_RE = re.compile(r"\\(begin|end)\{([^}]+)\}")
MISSING_PACKAGE_PATTERNS: Dict[str, str] = {
    r"\\includegraphics": "graphicx",
    r"\\mathbb": "amsfonts",
    r"\\begin\{align": "amsmath",
    r"\\begin\{cases": "amsmath",
    r"\\toprule": "booktabs",
    r"\\cite": "natbib",
}
SEMANTIC_WARNINGS = [
    (re.compile(r"\\\\"), "double-backslash-outside-math"),
    (re.compile(r"\$\$"), "double-dollar-math"),
]


@dataclass
class RepairRecommendation:
    code: str
    detail: str


@dataclass
class KimiK2LatexRepair:
    """Diagnose and repair LaTeX issues using heuristics + Kimi."""

    max_tokens: int = 320
    temperature: float = 0.0
    grammar: Optional[str] = LATEX_VALIDATION_GRAMMAR
    _adapter_cached: object | None = field(default=None, init=False, repr=False)
    _adapter_failed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        env_temp = os.environ.get("LATEXIFY_KIMI_K2_TEMPERATURE")
        if env_temp:
            try:
                self.temperature = float(env_temp)
            except ValueError:
                LOGGER.debug("Invalid LATEXIFY_KIMI_K2_TEMPERATURE=%s; using %s", env_temp, self.temperature)
        env_tokens = os.environ.get("LATEXIFY_KIMI_K2_MAX_TOKENS")
        if env_tokens:
            try:
                parsed = int(env_tokens)
                if parsed > 0:
                    self.max_tokens = parsed
            except ValueError:
                LOGGER.debug("Invalid LATEXIFY_KIMI_K2_MAX_TOKENS=%s; using %s", env_tokens, self.max_tokens)

    def preflight_check(self, tex_path: Path) -> Dict[str, object]:
        report = {
            "balanced": False,
            "packages": [],
        }
        if not tex_path.exists():
            return report
        text = tex_path.read_text(encoding="utf-8")
        balanced = self._balance_environments(text)
        if balanced != text:
            tex_path.write_text(balanced, encoding="utf-8")
            report["balanced"] = True
        packages = self._resolve_packages(text)
        if packages:
            self._inject_packages(tex_path, packages)
            report["packages"] = packages
        return report

    def diagnose_log(self, log_text: str) -> List[RepairRecommendation]:
        recommendations: List[RepairRecommendation] = []
        lowered = log_text.lower()
        if "undefined control sequence" in lowered:
            recommendations.append(RepairRecommendation(code="undefined-control", detail="Add macro or package"))
        if "missing $ inserted" in lowered:
            recommendations.append(RepairRecommendation(code="math-delimiter", detail="Balance math mode"))
        if "File `" in log_text and "not found" in log_text:
            recommendations.append(RepairRecommendation(code="missing-file", detail="Ensure assets/pdfs exist"))
        if "There were undefined references" in log_text or "Citation" in log_text:
            recommendations.append(RepairRecommendation(code="undefined-reference", detail="Run bibtex or add \nocite{}"))
        if "! LaTeX Error: Environment" in log_text and "undefined" in log_text:
            recommendations.append(RepairRecommendation(code="environment", detail="Load appropriate package"))
        return recommendations

    def semantic_validate(self, tex_path: Path) -> Dict[str, List[str]]:
        if not tex_path.exists():
            return {"issues": []}
        text = tex_path.read_text(encoding="utf-8")
        issues = []
        for pattern, code in SEMANTIC_WARNINGS:
            if pattern.search(text):
                issues.append(code)
        return {"issues": issues}

    def repair_from_log(self, tex_path: Path, log_path: Path) -> Dict[str, object]:
        summary: Dict[str, object] = {"actions": []}
        if not log_path.exists() or not tex_path.exists():
            return summary
        log_text = log_path.read_text(errors="ignore")
        recommendations = self.diagnose_log(log_text)
        if not recommendations:
            return summary
        adapter = self._adapter()
        if adapter is None:
            return summary
        snippet = tex_path.read_text(encoding="utf-8")
        prompt = self._render_prompt(snippet, log_text)
        suggestion = ""
        start = perf_counter()
        try:
            suggestion = adapter.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                grammar=self.grammar,
            )
            kimi_metrics.record_inference(perf_counter() - start, bool(suggestion))
        except Exception as exc:  # pragma: no cover - llama runtime
            LOGGER.debug("Kimi repair failed: %s", exc)
            kimi_metrics.record_inference(perf_counter() - start, False)
        if suggestion:
            cleaned = self._strip_tags(suggestion)
            if cleaned.strip():
                tex_path.write_text(cleaned, encoding="utf-8")
                summary["actions"].append({"type": "kimi-fix", "length": len(cleaned)})
                kimi_metrics.record_repair(True)
            else:
                kimi_metrics.record_repair(False)
        packages = self._resolve_packages(log_text)
        if packages:
            self._inject_packages(tex_path, packages)
            summary["actions"].append({"type": "add-packages", "packages": packages})
        if not summary.get("actions"):
            kimi_metrics.record_repair(False)
        return summary

    def _adapter(self):
        if self._adapter_failed:
            return None
        if self._adapter_cached is None:
            self._adapter_cached = get_kimi_adapter()
            if self._adapter_cached is None:
                self._adapter_failed = True
        return self._adapter_cached

    def _render_prompt(self, tex: str, log_text: str) -> str:
        log_excerpt = log_text.splitlines()[-40:]
        log_payload = "\n".join(log_excerpt)
        return (
            "You are a meticulous LaTeX repair agent."
            "Use the compilation log to fix the provided document."
            "Respond with updated LaTeX between <latex> tags.\n"
            f"<log>\n{log_payload}\n</log>\n"
            f"<latex>{tex}</latex>"
        )

    def _strip_tags(self, payload: str) -> str:
        text = payload.strip()
        if text.lower().startswith("<latex>"):
            text = text[7:]
        if "</latex>" in text:
            text = text.split("</latex>", 1)[0]
        return text.strip()

    def _balance_environments(self, tex: str) -> str:
        stack: List[str] = []
        corrections: List[str] = []
        for match in UNBALANCED_ENV_RE.finditer(tex):
            cmd, name = match.groups()
            if cmd == "begin":
                stack.append(name)
            else:
                if stack and stack[-1] == name:
                    stack.pop()
                else:
                    corrections.append(f"Missing \\begin{{{name}}}")
        for leftover in reversed(stack):
            tex += f"\n\\end{{{leftover}}}"
        if corrections:
            LOGGER.debug("Balanced environments by appending %s", corrections)
        return tex

    def _resolve_packages(self, text: str) -> List[str]:
        packages: List[str] = []
        for pattern, package in MISSING_PACKAGE_PATTERNS.items():
            if re.search(pattern, text) and package not in packages:
                packages.append(package)
        return packages

    def suggest_packages(self, payload: str) -> List[str]:
        return self._resolve_packages(payload)

    def _inject_packages(self, tex_path: Path, packages: List[str]) -> None:
        if not packages:
            return
        text = tex_path.read_text(encoding="utf-8")
        insert_idx = text.find("\\begin{document}")
        if insert_idx == -1:
            insert_idx = len(text)
        before = text[:insert_idx]
        after = text[insert_idx:]
        for package in packages:
            if f"\\usepackage{{{package}}}" not in before:
                before += f"\\usepackage{{{package}}}\n"
        tex_path.write_text(before + after, encoding="utf-8")


__all__ = ["KimiK2LatexRepair", "RepairRecommendation"]
