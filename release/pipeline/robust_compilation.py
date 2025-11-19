"""Incremental LaTeX compilation with error recovery and caching."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .latex_repair_agent import KimiK2LatexRepair

LOGGER = logging.getLogger(__name__)

SECTION_SPLIT_RE = re.compile(r"^(\\section\*?\{[^}]+\})", re.MULTILINE)
ERROR_PATTERNS = {
    "missing-package": re.compile(r"No file `([^']+\.sty)'"),
    "undefined-control": re.compile(r"Undefined control sequence"),
    "environment-mismatch": re.compile(r"! LaTeX Error: (?:Missing|Extra) \"(?:\\begin|\\end)"),
    "undefined-reference": re.compile(r"There were undefined references"),
    "citation": re.compile(r"Citation `[^']+' on page"),
}


@dataclass
class SectionUnit:
    name: str
    body: str
    start_line: int
    end_line: int
    checksum: str
    order: int


@dataclass
class IncrementalCompilationResult:
    success: bool
    pdf_path: Optional[Path]
    section_reports: List[Dict[str, object]]
    error_summary: List[Dict[str, object]]
    cache_path: Path
    log_path: Optional[Path]
    attempt_history: List[Dict[str, object]]
    recovery_success: bool


class LaTeXErrorClassifier:
    def classify(self, log_text: str) -> List[Dict[str, str]]:
        findings: List[Dict[str, str]] = []
        lowered = log_text.lower()
        for code, pattern in ERROR_PATTERNS.items():
            if pattern.search(log_text):
                findings.append({"code": code, "detail": pattern.pattern})
        if "Package" in log_text and "Warning" in log_text and "Reference" in log_text:
            findings.append({"code": "reference-warning", "detail": "reference warning"})
        if "Overfull \hbox" in log_text:
            findings.append({"code": "layout", "detail": "overfull-hbox"})
        if "Fatal error" in lowered or "Emergency stop" in log_text:
            findings.append({"code": "fatal", "detail": "compiler-stopped"})
        return findings


class CompilationCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_path = cache_dir / "robust_compile_cache.json"
        self.data: Dict[str, object] = {"preamble": "", "sections": {}}
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            try:
                self.data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                LOGGER.debug("Corrupt compilation cache; resetting")

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def preamble_changed(self, checksum: str) -> bool:
        return self.data.get("preamble") != checksum

    def update_preamble(self, checksum: str) -> None:
        self.data["preamble"] = checksum

    def section_changed(self, unit: SectionUnit) -> bool:
        sections: Dict[str, str] = self.data.setdefault("sections", {})  # type: ignore[assignment]
        return sections.get(unit.name) != unit.checksum

    def update_section(self, unit: SectionUnit) -> None:
        sections: Dict[str, str] = self.data.setdefault("sections", {})  # type: ignore[assignment]
        sections[unit.name] = unit.checksum


class RobustCompiler:
    def __init__(
        self,
        tex_path: Path,
        *,
        cache_dir: Optional[Path] = None,
        repair_agent: Optional[KimiK2LatexRepair] = None,
        progressive_preamble: bool = True,
        max_retries: int = 2,
    ) -> None:
        self.tex_path = tex_path
        self.project_dir = tex_path.parent
        self.cache = CompilationCache(cache_dir or tex_path.parent / ".latexify_cache")
        self.repair_agent = repair_agent or KimiK2LatexRepair()
        self.classifier = LaTeXErrorClassifier()
        self.progressive_preamble = progressive_preamble
        self.staging_dir = self.cache.cache_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max(1, max_retries)
        self._attempt_history: List[Dict[str, object]] = []
        self._recovery_success = False

    def run(self) -> IncrementalCompilationResult:
        section_reports: List[Dict[str, object]] = []
        preamble, body = self._split_document()
        preamble_hash = self._hash_text(preamble)
        if self.cache.preamble_changed(preamble_hash):
            self.cache.update_preamble(preamble_hash)
        units = self._extract_units(body)
        for idx, unit in enumerate(units):
            needs_compile = self.cache.section_changed(unit)
            if not needs_compile:
                section_reports.append({"section": unit.name, "compiled": False, "status": "cached"})
                continue
            success = self._compile_unit(unit, preamble)
            section_reports.append({"section": unit.name, "compiled": True, "success": success})
            if success:
                self.cache.update_section(unit)
            else:
                break
        self.cache.save()
        final_result = self._compile_full_document()
        error_summary: List[Dict[str, object]] = []
        log_path: Optional[Path] = None
        recovery_attempted = False
        if not final_result:
            log_path = self.tex_path.with_suffix(".log")
            if log_path.exists():
                error_summary = self.classifier.classify(log_path.read_text(errors="ignore"))
                recovery_attempted = True
                self.repair_agent.repair_from_log(self.tex_path, log_path)
                final_result = self._compile_full_document(fallback=True)
                self._recovery_success = final_result is not None
            else:
                self._recovery_success = False
        else:
            self._recovery_success = False
        success = final_result is not None and final_result.exists()
        return IncrementalCompilationResult(
            success=success,
            pdf_path=final_result if success else None,
            section_reports=section_reports,
            error_summary=error_summary,
            cache_path=self.cache.cache_path,
            log_path=self.tex_path.with_suffix(".log") if self.tex_path.with_suffix(".log").exists() else None,
            attempt_history=list(self._attempt_history),
            recovery_success=recovery_attempted and self._recovery_success,
        )

    def _split_document(self) -> Tuple[str, str]:
        text = self.tex_path.read_text(encoding="utf-8")
        marker = "\\begin{document}"
        idx = text.find(marker)
        if idx == -1:
            return "", text
        return text[: idx + len(marker)], text[idx + len(marker) :]

    def _extract_units(self, body: str) -> List[SectionUnit]:
        if not body.strip():
            return [SectionUnit(name="document", body=body, start_line=1, end_line=len(body.splitlines()), checksum=self._hash_text(body), order=0)]
        segments = SECTION_SPLIT_RE.split(body)
        units: List[SectionUnit] = []
        current = []
        current_name = "section-000"
        line_counter = 0
        order = 0
        for segment in segments:
            if SECTION_SPLIT_RE.match(segment):
                if current:
                    body_text = "".join(current)
                    units.append(
                        SectionUnit(
                            name=current_name,
                            body=body_text,
                            start_line=line_counter,
                            end_line=line_counter + len(body_text.splitlines()),
                            checksum=self._hash_text(body_text),
                            order=order,
                        )
                    )
                    order += 1
                    line_counter += len(body_text.splitlines())
                current_name = f"section-{order:03d}"
                current = [segment]
            else:
                current.append(segment)
        if current:
            body_text = "".join(current)
            units.append(
                SectionUnit(
                    name=current_name,
                    body=body_text,
                    start_line=line_counter,
                    end_line=line_counter + len(body_text.splitlines()),
                    checksum=self._hash_text(body_text),
                    order=order,
                )
            )
        return units

    def _compile_unit(self, unit: SectionUnit, preamble: str) -> bool:
        staged = self.staging_dir / f"{unit.name}.tex"
        staged.write_text(self._assemble_temp(preamble, unit.body), encoding="utf-8")
        log_path = staged.with_suffix(".log")
        if log_path.exists():
            log_path.unlink()
        ok, _ = self._run_engine(staged)
        if not ok and log_path.exists():
            self._attempt_recovery(staged, log_path)
            ok, _ = self._run_engine(staged)
        if not ok:
            LOGGER.warning("Section %s failed to compile", unit.name)
        return ok

    def _compile_full_document(self, fallback: bool = False) -> Optional[Path]:
        if fallback:
            LOGGER.info("Running fallback compilation strategy")
        pdf_path = self.tex_path.with_suffix(".pdf")
        if pdf_path.exists():
            pdf_path.unlink()
        compiled_path, success = self._run_compile_with_retries(self.tex_path, fallback=fallback)
        return compiled_path if success else None

    def _run_compile_with_retries(self, tex_file: Path, *, fallback: bool) -> tuple[Optional[Path], bool]:
        pdf_path = tex_file.with_suffix(".pdf")
        for _ in range(self.max_retries):
            ok, engine = self._run_engine(tex_file, fallback=fallback)
            record = {
                "attempt": len(self._attempt_history) + 1,
                "engine": engine or "unknown",
                "target": tex_file.name,
                "fallback": fallback,
                "success": ok,
            }
            self._attempt_history.append(record)
            if ok and pdf_path.exists():
                return pdf_path, True
        return (pdf_path if pdf_path.exists() else None), False

    def _attempt_recovery(self, staged: Path, log_path: Path) -> None:
        log_text = log_path.read_text(errors="ignore")
        diagnostics = self.classifier.classify(log_text)
        if not diagnostics:
            return
        if any(item["code"] == "environment-mismatch" for item in diagnostics):
            repaired = self.repair_agent.preflight_check(staged)
            LOGGER.debug("Applied environment balancing on %s: %s", staged.name, repaired)
        if any(item["code"] == "missing-package" for item in diagnostics):
            packages = self.repair_agent.suggest_packages(log_text)
            self.repair_agent._inject_packages(staged, packages)

    def _assemble_temp(self, preamble: str, body: str) -> str:
        doc = preamble
        doc += "\n% -- incremental compile --\n"
        doc += body
        if "\\end{document}" not in doc:
            doc += "\n\\end{document}\n"
        return doc

    def _run_engine(self, tex_file: Path, *, fallback: bool = False) -> tuple[bool, Optional[str]]:
        engines = ["tectonic", "latexmk", "pdflatex"]
        if fallback:
            engines.reverse()
        for engine in engines:
            binary = shutil.which(engine)
            if not binary:
                continue
            cmd = self._build_command(engine, tex_file)
            try:
                subprocess.run(cmd, check=True, cwd=str(tex_file.parent))
                return True, engine
            except subprocess.CalledProcessError as exc:
                LOGGER.debug("%s failed for %s: %s", engine, tex_file.name, exc)
                continue
        LOGGER.warning("No LaTeX engine succeeded for %s", tex_file.name)
        return False, None

    def _build_command(self, engine: str, tex_file: Path) -> List[str]:
        if engine == "tectonic":
            return [engine, "--keep-logs", "--keep-intermediates", str(tex_file)]
        if engine == "latexmk":
            return [engine, "-pdf", "-interaction=nonstopmode", str(tex_file)]
        return [engine, "-interaction=nonstopmode", "-halt-on-error", tex_file.name]

    def _hash_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()


def run_robust_compilation(
    tex_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    repair_agent: Optional[KimiK2LatexRepair] = None,
    max_retries: Optional[int] = None,
) -> IncrementalCompilationResult:
    compiler = RobustCompiler(
        tex_path,
        cache_dir=cache_dir,
        repair_agent=repair_agent,
        max_retries=max_retries or 2,
    )
    return compiler.run()


__all__ = [
    "run_robust_compilation",
    "IncrementalCompilationResult",
    "RobustCompiler",
]
