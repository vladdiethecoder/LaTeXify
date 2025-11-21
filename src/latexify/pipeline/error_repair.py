"""LLM-assisted LaTeX fixer that parses compilation logs."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..models.local_llm import LocalLLMConfig, LocalLLMOrchestrator, LocalLLMUnavailable

try:  # pragma: no cover - optional dependency
    from texoutparse import parser as tex_parser  # type: ignore
except Exception:  # pragma: no cover
    tex_parser = None  # type: ignore

LOGGER = logging.getLogger(__name__)
ERROR_LINE_RE = re.compile(r"l\.(\d+)")
DEFAULT_PROMPT = (
    "You are a LaTeX repair assistant. Given the compiler log excerpt describing the error and "
    "the LaTeX snippet that triggered it, output a corrected LaTeX snippet that fixes the issue "
    "without removing valid math content. Respond ONLY with the fixed snippet wrapped inside "
    "<latex>...</latex>."
)


@dataclass
class LaTeXErrorRepair:
    """Parse latexmk/tectonic logs and ask a local LLM to repair failing snippets."""

    max_passes: int = 3
    repair_context_lines: int = 6
    max_tokens: int = 384
    temperature: float = 0.1
    repair_prompt: str = DEFAULT_PROMPT
    model_path: str | None = None
    grammar_path: str | None = None
    _llm: LocalLLMOrchestrator | None = field(default=None, init=False, repr=False)
    _llm_failed: bool = field(default=False, init=False, repr=False)

    def analyze_log(self, log_path: Path) -> List[str]:
        if not log_path.exists():
            return []
        if tex_parser is not None:  # pragma: no cover - optional dependency
            try:
                doc = tex_parser.parse(log_path.read_text(errors="ignore"))
                if doc and doc.errors:
                    return [err.message.lower() for err in doc.errors]
            except Exception:
                LOGGER.debug("texoutparse failed to inspect %s", log_path, exc_info=True)
        text = log_path.read_text(errors="ignore")
        return [text]

    def _discover_model(self) -> str | None:
        search_root = Path(__file__).resolve().parents[1] / "models" / "llm"
        if not search_root.exists():
            return None
        for candidate in sorted(search_root.rglob("*.gguf")):
            if candidate.is_file():
                return str(candidate)
        return None

    def _ensure_llm(self) -> LocalLLMOrchestrator | None:
        if self._llm_failed:
            return None
        if self._llm is not None:
            return self._llm
        model_path = self.model_path or os.environ.get("LATEXIFY_REPAIR_LLM_MODEL")
        if not model_path:
            model_path = self._discover_model()
        if not model_path:
            self._llm_failed = True
            LOGGER.debug("No repair LLM model detected; skipping error repair.")
            return None
        grammar_path = self.grammar_path or os.environ.get("LATEXIFY_REPAIR_LLM_GRAMMAR")
        config = LocalLLMConfig(
            model_path=Path(model_path).expanduser(),
            grammar_path=Path(grammar_path).expanduser() if grammar_path else None,
        )
        try:
            self._llm = LocalLLMOrchestrator(config)
        except (LocalLLMUnavailable, OSError, RuntimeError) as exc:
            LOGGER.warning("LLM repair unavailable: %s", exc)
            self._llm_failed = True
            self._llm = None
        return self._llm

    def _extract_error_windows(self, log_text: str) -> List[Tuple[int, int, str]]:
        windows: List[Tuple[int, int, str]] = []
        for match in ERROR_LINE_RE.finditer(log_text):
            line = int(match.group(1))
            start = max(1, line - self.repair_context_lines)
            end = line + self.repair_context_lines
            span_start = max(0, match.start() - 400)
            span_end = min(len(log_text), match.end() + 400)
            excerpt = log_text[span_start:span_end].strip()
            windows.append((start, end, excerpt))
        return windows[: self.max_passes]

    def _apply_llm_patch(
        self,
        snippet: str,
        log_excerpt: str,
        engine: LocalLLMOrchestrator,
    ) -> str | None:
        if not snippet.strip():
            return None
        prompt = (
            f"{self.repair_prompt}\n\n<log>\n{log_excerpt.strip()}\n</log>\n"
            f"<source>\n{snippet.strip()}\n</source>\n<latex>"
        )
        try:
            output = engine.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # pragma: no cover - llama runtime
            LOGGER.debug("LLM repair failed: %s", exc)
            return None
        if not output:
            return None
        cleaned = output.strip()
        if cleaned.lower().startswith("<latex>"):
            cleaned = cleaned[7:]
        if "</latex>" in cleaned:
            cleaned = cleaned.split("</latex>", 1)[0]
        return cleaned.strip() or None

    def repair(self, tex_path: Path, log_path: Path) -> bool:
        if not log_path.exists() or not tex_path.exists():
            return False
        windows = self._extract_error_windows(log_path.read_text(errors="ignore"))
        if not windows:
            return False
        engine = self._ensure_llm()
        if engine is None:
            LOGGER.info("Skipping LLM repair because no local model is available.")
            return False
        lines = tex_path.read_text(encoding="utf-8").splitlines()
        changed = False
        for start, end, excerpt in windows:
            snippet = "\n".join(lines[start - 1 : min(len(lines), end)])
            suggestion = self._apply_llm_patch(snippet, excerpt, engine)
            if not suggestion:
                continue
            new_lines = suggestion.splitlines()
            lines[start - 1 : min(len(lines), end)] = new_lines
            changed = True
        if changed:
            tex_path.write_text("\n".join(lines), encoding="utf-8")
            LOGGER.info("Applied LLM-based LaTeX repairs for %s", tex_path.name)
        return changed


__all__ = ["LaTeXErrorRepair"]
