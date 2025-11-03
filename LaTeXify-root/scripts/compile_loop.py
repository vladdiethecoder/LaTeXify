#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_loop.py

Final Compilation Loop for LaTeXify.

Key updates (2025-11-01):
- Robust snippet path detection (absolute/relative) from log "(.../build/snippets/*.tex)".
- Macro-based fallback for "Undefined control sequence":
  * Extract undefined macro from error context (e.g., "\mystery" near "l.<N>").
  * Search uniquely under build/snippets for that macro; if found, derive file+line.
  * Hand precise target to auto-fix so main.tex is never edited by mistake.

Why:
- TeX logs often show "l.<N>" but the *responsible* file may be a snippet included elsewhere
  and the line number can be asynchronous/misaligned. Macro search is a practical
  disambiguation step. See references in docs/comments.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dev.eval.metrics import bertscore_f1 as _bertscore_f1
    from dev.eval.metrics import cer as _cer
    from dev.eval.metrics import wer as _wer
except Exception:  # pragma: no cover - optional dependency surface
    _bertscore_f1 = None
    _cer = None
    _wer = None

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # assumes this file is under <repo>/scripts/
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "dev" / "runs"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "build"
DEFAULT_MAIN_TEX = DEFAULT_BUILD_DIR / "main.tex"
DEFAULT_PLAN_PATH = DEFAULT_BUILD_DIR / "plan.json"
DEFAULT_SNIPPET_DIR = DEFAULT_BUILD_DIR / "snippets"
KB_LATEX_DIR = PROJECT_ROOT / "kb" / "latex"
AGGREGATOR_SCRIPT = HERE / "aggregator.py"

PROMOTE_WER_THRESHOLD = 0.35
PROMOTE_CER_THRESHOLD = 0.25

# JSON schema documentation for compile_metrics.json to keep analytics stable.
COMPILE_METRICS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "run_id",
        "generated_at",
        "compile_report",
        "compilation_success",
        "final_status",
        "passes",
        "auto_fix_attempts",
        "auto_fix_status",
        "pdf_artifact",
        "provenance",
        "promote_to_kb",
        "text_metrics",
    ],
    "properties": {
        "run_id": {"type": "string"},
        "generated_at": {"type": "string", "format": "date-time"},
        "compile_report": {"type": "string"},
        "compilation_success": {"type": "boolean"},
        "final_status": {"type": ["string", "null"]},
        "passes": {"type": ["integer", "null"]},
        "auto_fix_attempts": {"type": ["integer", "null"]},
        "auto_fix_status": {"type": ["string", "null"]},
        "pdf_artifact": {"type": ["string", "null"]},
        "provenance": {"type": ["string", "null"], "enum": ["rag", "rule", None]},
        "promote_to_kb": {"type": "boolean"},
        "text_metrics": {
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"type": "string"},
                "hypothesis_path": {"type": ["string", "null"]},
                "reference_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reference_candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "lengths": {
                    "type": "object",
                    "properties": {
                        "hypothesis_chars": {"type": "integer"},
                        "hypothesis_words": {"type": "integer"},
                        "reference_chars": {"type": "integer"},
                        "reference_words": {"type": "integer"},
                    },
                },
                "scores": {
                    "type": "object",
                    "properties": {
                        "cer": {"type": ["number", "null"]},
                        "wer": {"type": ["number", "null"]},
                        "bertscore_f1": {"type": ["number", "null"]},
                    },
                },
                "bertscore_available": {"type": "boolean"},
                "promotion_thresholds": {
                    "type": "object",
                    "properties": {
                        "wer": {"type": "number"},
                        "cer": {"type": "number"},
                    },
                },
                "warnings": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
}

# --- Error classifier: regex -> category -------------------------------------

ERROR_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"! Undefined control sequence\."), "undefined_control_sequence"),
    (re.compile(r"LaTeX Error: File `([^`]+)` not found\."), "file_not_found"),
    (re.compile(r"! I can't find file `([^']+)'\."), "file_not_found"),
    (re.compile(r"LaTeX Error: File .*\.sty' not found"), "missing_package"),
    (re.compile(r"LaTeX Error: .* ended by \\end\{([^\}]+)\}"), "env_mismatch"),
    (re.compile(r"Runaway argument\?"), "runaway_argument"),
    (re.compile(r"Emergency stop\."), "emergency_stop"),
    (re.compile(r"Reference .* undefined"), "ref_undefined"),
    (re.compile(r"Citation `[^']+' on page .* undefined"), "citation_undefined"),
    (re.compile(r"BibTeX"), "bibtex_issue"),
    (re.compile(r"makeindex"), "index_issue"),
]

# Capture any snippet path inside parentheses, abs or rel:
# e.g. "(/abs/path/.../build/snippets/0001.tex)" or "(./build/snippets/0001.tex)"
SNIPPET_IN_PARENS = re.compile(r"\(([^()]*build[/\\]snippets[^()]*\.tex)\)")
LINE_HINT = re.compile(r"\bl\.(\d+)\b")  # TeX prints "l.<num>"
MACRO_NEARBY = re.compile(r"(\\[A-Za-z@]+)")  # best effort capture of control seq (e.g., \mystery)


@dataclass
class CompileError:
    category: str
    message: str
    file: Optional[str]
    line: Optional[int]
    code_excerpt: Optional[str]


def _read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        data = path.read_bytes()
        return data[:max_bytes].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_excerpt(source: Path, line_no: Optional[int], context: int = 2) -> Optional[str]:
    if not source or not source.exists() or line_no is None or line_no < 1:
        return None
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        i = line_no - 1
        start = max(0, i - context)
        end = min(len(lines), i + context + 1)
        excerpt = "\n".join(f"{idx+1:>4}: {lines[idx]}" for idx in range(start, end))
        return excerpt
    except Exception:
        return None


def _which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / cmd
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _compiler_command(main_tex: Path) -> List[str]:
    latexmk = _which("latexmk")
    if latexmk:
        return [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", str(main_tex)]
    pdflatex = _which("pdflatex") or "pdflatex"
    return [pdflatex, "-interaction=nonstopmode", "-halt-on-error", str(main_tex)]


def _run_compile(cmd: List[str], workdir: Path, stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(cmd, cwd=str(workdir), stdout=out, stderr=err)
        return proc.wait()


def _tail(text: str, n: int = 200) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _load_source_map(build_dir: Path) -> Dict:
    path = build_dir / "main.sourcemap.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_path(build_dir: Path, candidate: Optional[str]) -> Optional[Path]:
    if not candidate:
        return None
    p = Path(candidate)
    if not p.is_absolute():
        p = (build_dir / candidate).resolve()
    return p


def _candidate_keys(path: Path, build_dir: Path) -> List[str]:
    keys = {path.as_posix()}
    try:
        rel = path.relative_to(build_dir)
        keys.add(rel.as_posix())
    except ValueError:
        pass
    return list(keys)


def _resolve_with_source_map(
    compile_error: Optional[CompileError],
    source_map: Dict,
    build_dir: Path,
) -> Tuple[Optional[CompileError], Optional[Path], Optional[Path]]:
    if not compile_error:
        return None, None, None

    entries = source_map.get("entries") or []
    sections = source_map.get("sections") or {}
    snippet_path: Optional[Path] = None
    section_path: Optional[Path] = None

    if compile_error.file:
        file_path = Path(compile_error.file)
        keys = _candidate_keys(file_path, build_dir)
        for key in keys:
            info = sections.get(key)
            if info:
                section_path = _resolve_path(build_dir, key)
                snippet_path = _resolve_path(build_dir, info.get("snippet")) if info.get("snippet") else None
                if snippet_path:
                    new_err = replace(compile_error, file=str(snippet_path))
                    return new_err, snippet_path, section_path
        if any(key.endswith("main.tex") for key in keys):
            line_no = compile_error.line
            if isinstance(line_no, int):
                for entry in entries:
                    main_info = entry.get("main") or {}
                    start = main_info.get("start")
                    end = main_info.get("end")
                    if isinstance(start, int) and isinstance(end, int) and start <= line_no <= end:
                        snippet_path = _resolve_path(build_dir, entry.get("snippet_file"))
                        section_path = _resolve_path(build_dir, entry.get("section_file"))
                        if snippet_path:
                            new_line = 1
                            new_err = replace(compile_error, file=str(snippet_path), line=new_line)
                            return new_err, snippet_path, section_path
    else:
        line_no = compile_error.line
        if isinstance(line_no, int):
            for entry in entries:
                main_info = entry.get("main") or {}
                start = main_info.get("start")
                end = main_info.get("end")
                if isinstance(start, int) and isinstance(end, int) and start <= line_no <= end:
                    snippet_path = _resolve_path(build_dir, entry.get("snippet_file"))
                    section_path = _resolve_path(build_dir, entry.get("section_file"))
                    if snippet_path:
                        new_err = replace(compile_error, file=str(snippet_path), line=1)
                        return new_err, snippet_path, section_path

    return compile_error, snippet_path, section_path


def _rerun_aggregator(plan_path: Path, snippets_dir: Path, build_dir: Path) -> Dict:
    if not AGGREGATOR_SCRIPT.exists():
        return {"status": "skipped", "reason": "aggregator.py missing"}
    if not plan_path.exists():
        return {"status": "skipped", "reason": f"plan not found: {plan_path}"}
    cmd = [
        sys.executable,
        str(AGGREGATOR_SCRIPT),
        "--plan",
        str(plan_path),
        "--snippets_dir",
        str(snippets_dir),
        "--out_dir",
        str(build_dir),
        "--no_compile",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    status = "ok" if proc.returncode == 0 else "error"
    return {
        "status": status,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def _word_count(text: str) -> int:
    return len(text.split())


def _read_reference_directory(dir_path: Path) -> Tuple[str, List[Path]]:
    if not dir_path.exists() or not dir_path.is_dir():
        return "", []
    files = sorted({*dir_path.glob("*.md"), *dir_path.glob("*.txt")})
    texts: List[str] = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            content = ""
        if content:
            texts.append(content)
    combined = "\n\n".join(texts).strip()
    return combined, files


def _gather_reference_corpus(
    runs_root: Path,
    run_id: str,
    provenance: str,
) -> Tuple[str, List[Path], List[Path]]:
    outputs_dir = runs_root / run_id / "outputs"
    if not outputs_dir.exists():
        return "", [], [outputs_dir]

    if provenance == "rag":
        preference = ["rag", "fallback", "rule"]
    else:
        preference = ["rule", "fallback", "rag"]

    tried: List[Path] = []
    seen: set[str] = set()
    for label in preference:
        candidate = outputs_dir / label
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        tried.append(candidate)
        text, files = _read_reference_directory(candidate)
        if text:
            return text, files, tried

    # As a last resort, check the outputs directory directly.
    tried.append(outputs_dir)
    text, files = _read_reference_directory(outputs_dir)
    if text:
        return text, files, tried
    return "", files, tried


def _compute_text_metrics(
    main_tex: Path,
    hyp_text: str,
    reference_text: str,
    reference_sources: List[Path],
    candidate_dirs: List[Path],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "ok",
        "hypothesis_path": str(main_tex),
        "reference_sources": [str(p) for p in reference_sources],
        "reference_candidates": [str(p) for p in candidate_dirs],
        "lengths": {
            "hypothesis_chars": len(hyp_text),
            "hypothesis_words": _word_count(hyp_text),
            "reference_chars": len(reference_text),
            "reference_words": _word_count(reference_text),
        },
        "scores": {
            "cer": None,
            "wer": None,
            "bertscore_f1": None,
        },
        "bertscore_available": False,
        "promotion_thresholds": {
            "wer": PROMOTE_WER_THRESHOLD,
            "cer": PROMOTE_CER_THRESHOLD,
        },
    }
    warnings: List[str] = []

    if _cer is not None:
        try:
            result["scores"]["cer"] = float(_cer(hyp_text, reference_text))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"cer_failed: {exc}")
            result["scores"]["cer"] = None
    else:
        warnings.append("cer_unavailable")

    if _wer is not None:
        try:
            result["scores"]["wer"] = float(_wer(hyp_text, reference_text))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"wer_failed: {exc}")
            result["scores"]["wer"] = None
    else:
        warnings.append("wer_unavailable")

    bert_f1: Optional[float] = None
    if _bertscore_f1 is not None and hyp_text.strip() and reference_text.strip():
        try:
            bert_val = _bertscore_f1(hyp_text, reference_text)
            if bert_val is not None:
                bert_f1 = float(bert_val)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"bertscore_failed: {exc}")
            bert_f1 = None
    result["scores"]["bertscore_f1"] = bert_f1
    result["bertscore_available"] = bert_f1 is not None

    if warnings:
        result["warnings"] = warnings

    return result


def _compute_text_metrics_view(
    build_dir: Path,
    runs_root: Path,
    run_id: str,
    status: Optional[str],
    provenance: str,
) -> Dict[str, Any]:
    main_tex = build_dir / "main.tex"
    base: Dict[str, Any] = {
        "status": "compile_failed" if status != "ok" else "missing_main_tex",
        "hypothesis_path": str(main_tex),
        "reference_sources": [],
        "reference_candidates": [],
        "lengths": {},
        "scores": {},
        "bertscore_available": False,
        "promotion_thresholds": {
            "wer": PROMOTE_WER_THRESHOLD,
            "cer": PROMOTE_CER_THRESHOLD,
        },
    }

    if status != "ok":
        return base

    if not main_tex.exists():
        return base

    hyp_text = _read_text(main_tex)
    reference_text, sources, candidates = _gather_reference_corpus(runs_root, run_id, provenance)
    base["reference_sources"] = [str(p) for p in sources]
    base["reference_candidates"] = [str(p) for p in candidates]
    base["lengths"] = {
        "hypothesis_chars": len(hyp_text),
        "hypothesis_words": _word_count(hyp_text),
        "reference_chars": len(reference_text),
        "reference_words": _word_count(reference_text),
    }

    if not reference_text.strip():
        base["status"] = "reference_unavailable"
        return base

    return _compute_text_metrics(main_tex, hyp_text, reference_text, sources, candidates)


def _infer_provenance(build_dir: Path) -> str:
    snippets_dir = build_dir / "snippets"
    if snippets_dir.exists():
        for tex_file in snippets_dir.rglob("*.tex"):
            if tex_file.is_file():
                return "rag"
    return "rule"


def _should_promote(report: Dict[str, Any], text_metrics: Dict[str, Any]) -> bool:
    if report.get("status") != "ok":
        return False
    if report.get("provenance") != "rag":
        return False
    if text_metrics.get("status") != "ok":
        return False
    scores = text_metrics.get("scores") or {}
    wer_score = scores.get("wer")
    cer_score = scores.get("cer")
    if wer_score is None or cer_score is None:
        return False
    return wer_score <= PROMOTE_WER_THRESHOLD and cer_score <= PROMOTE_CER_THRESHOLD


def _sanitize_text_metrics(data: Any) -> Any:
    if data is None:
        return {"status": "unavailable"}
    try:
        return json.loads(json.dumps(data, ensure_ascii=False, default=str))
    except Exception:  # pragma: no cover - defensive
        return {"status": "unavailable"}


def _build_metrics_payload(run_id: str, report_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compile_report": str(report_path),
        "compilation_success": report.get("status") == "ok",
        "final_status": report.get("status"),
        "passes": report.get("passes"),
        "auto_fix_attempts": int(report.get("retry_policy", {}).get("auto_fix_attempted", False)),
        "auto_fix_status": (report.get("auto_fix") or {}).get("status"),
        "pdf_artifact": (report.get("artifacts") or {}).get("pdf_file"),
        "provenance": report.get("provenance"),
        "promote_to_kb": bool(report.get("promote_to_kb", False)),
        "text_metrics": _sanitize_text_metrics(report.get("text_metrics")),
    }
    return payload


def _log_successful_fix(
    build_dir: Path,
    snippet_path: Optional[Path],
    before: Optional[str],
    after: Optional[str],
    compile_error: CompileError,
    fix_payload: Dict,
) -> None:
    if not snippet_path or before is None or after is None:
        return
    log_path = build_dir / "successful_fixes.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snippet": str(snippet_path),
        "error": {
            "category": compile_error.category,
            "message": compile_error.message,
            "line": compile_error.line,
        },
        "before": before,
        "after": after,
        "auto_fix": fix_payload,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
def _parse_first_error_and_hints(log_text: str, project_root: Path) -> Tuple[Optional[CompileError], Optional[int]]:
    """
    Return first error (category/message/file/line/excerpt) + the index of the 'Undefined control sequence'
    line, so we can mine nearby macro tokens if needed.
    """
    first_category = None
    first_message = None
    first_file = None
    first_line = None
    undefined_idx = None

    lines = log_text.splitlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()

        for patt, cat in ERROR_PATTERNS:
            if patt.search(line):
                if first_category is None:
                    first_category = cat
                    first_message = line
                    if cat == "undefined_control_sequence":
                        undefined_idx = idx

        # Prefer explicit snippet paths if present
        mfile = SNIPPET_IN_PARENS.search(line)
        if mfile and first_file is None:
            cand = (project_root / mfile.group(1)).resolve()
            # keep only if inside project
            try:
                if project_root in cand.parents or cand == project_root:
                    first_file = str(cand)
            except Exception:
                pass

        mline = LINE_HINT.search(line)
        if mline and first_line is None:
            try:
                first_line = int(mline.group(1))
            except ValueError:
                pass

        if first_category and first_message and (first_file or first_line is not None):
            break

    if not first_category and not first_message:
        return None, None

    ce = CompileError(
        category=first_category or "unknown_error",
        message=first_message or "Compilation failed (unknown cause).",
        file=first_file,
        line=first_line,
        code_excerpt=None,
    )
    return ce, undefined_idx


def _guess_snippet_from_macro(
    log_text: str,
    undefined_idx: Optional[int],
    build_dir: Path,
) -> Tuple[Optional[Path], Optional[int], Optional[str]]:
    """
    Heuristic for 'Undefined control sequence' with missing snippet path.
    - Look within a small window around the undefined message to extract a control sequence token.
    - Search uniquely under build/snippets for that token.
    - Return (file_path, line_no, macro_name).
    """
    if undefined_idx is None:
        return None, None, None

    lines = log_text.splitlines()
    window = lines[max(0, undefined_idx - 4): min(len(lines), undefined_idx + 6)]
    macro = None

    # Try to find a macro token near the 'l.<N>' context
    for w in window:
        if "l." in w or "Undefined control sequence" in w:
            m = MACRO_NEARBY.search(w)
            if m:
                macro = m.group(1)  # includes backslash
                break

    if not macro:
        return None, None, None

    macro_plain = macro[1:]  # strip backslash for searching
    snippets_root = build_dir / "snippets"
    if not snippets_root.exists():
        return None, None, macro

    matches: List[Tuple[Path, int]] = []
    for tex in snippets_root.rglob("*.tex"):
        try:
            text = tex.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # find first occurrence to get a reasonable line number
        pos = text.find(macro)
        if pos == -1:
            # also try the plain name with leading backslash inserted at runtime
            pos = text.find("\\" + macro_plain)
        if pos != -1:
            # compute line
            line_no = text[:pos].count("\n") + 1
            matches.append((tex, line_no))
    if len(matches) == 1:
        return matches[0][0], matches[0][1], macro
    return None, None, macro


def compile_once(
    main_tex: Path,
    build_dir: Path,
    run_dir: Path,
) -> Tuple[int, str, str, Path]:
    cmd = _compiler_command(main_tex)
    compiler_name = Path(cmd[0]).name
    stdout_path = run_dir / f"{compiler_name}.stdout.txt"
    stderr_path = run_dir / f"{compiler_name}.stderr.txt"
    exit_code = _run_compile(cmd, workdir=build_dir, stdout_path=stdout_path, stderr_path=stderr_path)

    log_candidates = sorted(build_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    log_path = log_candidates[0] if log_candidates else (run_dir / "compile.log")
    log_text = _read_text(log_path) if log_candidates else (_read_text(stdout_path) + "\n" + _read_text(stderr_path))
    return exit_code, log_text, compiler_name, log_path


def write_report(report_path: Path, report: Dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_metrics(build_dir: Path, run_id: str, report_path: Path, report: Dict) -> Path:
    """Persist a light-weight metrics view alongside the full report.

    The serialized structure adheres to :data:`COMPILE_METRICS_SCHEMA` so downstream
    analytics jobs can rely on a stable schema.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    metrics = _build_metrics_payload(run_id, report_path, report)
    metrics_path = build_dir / "compile_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics_path


def invoke_auto_fix(
    compile_error: CompileError,
    project_root: Path,
    build_dir: Path,
    run_dir: Path,
    seed: int,
    snippet_path: Optional[Path],
    section_path: Optional[Path],
    log_path: Optional[Path],
    kb_root: Path,
) -> Dict:
    fixer = HERE / "auto_fix.py"
    if not fixer.exists():
        return {"status": "skipped", "reason": "auto_fix.py not found"}

    args = [
        sys.executable, str(fixer),
        "--error-category", compile_error.category,
        "--message", compile_error.message,
        "--run-dir", str(run_dir),
        "--project-root", str(project_root),
        "--build-dir", str(build_dir),
        "--seed", str(seed),
    ]
    if compile_error.file:
        args += ["--file", compile_error.file]
    if compile_error.line is not None:
        args += ["--line", str(compile_error.line)]
    if snippet_path:
        args += ["--snippet", str(snippet_path)]
    if section_path:
        args += ["--section", str(section_path)]
    if log_path and log_path.exists():
        args += ["--log-path", str(log_path)]
    if kb_root.exists():
        args += ["--kb-root", str(kb_root)]

    proc = subprocess.run(args, capture_output=True, text=True)
    out = proc.stdout.strip()
    try:
        result = json.loads(out)
    except Exception:
        result = {"status": "error", "raw_stdout": out, "stderr": proc.stderr, "returncode": proc.returncode}
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LaTeXify Final Compilation Loop")
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX, help="Path to main.tex")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="Build directory containing main.tex")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Root for run artifacts (/dev/runs by default)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id (default: timestamp)")
    parser.add_argument("--auto-fix", type=int, default=1, help="Attempt self-correction once on failure (0/1)")
    parser.add_argument("--max-retries", type=int, default=1, help="Bounded retry policy (only used post-fix)")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed to record")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH, help="Path to plan.json for re-aggregation")
    parser.add_argument("--snippets-dir", type=Path, default=DEFAULT_SNIPPET_DIR, help="Directory containing snippet tex files")
    parser.add_argument("--kb-root", type=Path, default=KB_LATEX_DIR, help="Knowledge base root for auto-fix context")
    args = parser.parse_args(argv)

    project_root = PROJECT_ROOT
    build_dir = args.build_dir.resolve()
    main_tex = args.main_tex.resolve()
    runs_root = args.runs_root.resolve()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    seed = int(args.seed) if args.seed is not None else 1337
    plan_path = args.plan.resolve()
    snippets_dir = args.snippets_dir.resolve()
    kb_root = args.kb_root.resolve()

    run_dir = runs_root / run_id / "compile"
    run_dir.mkdir(parents=True, exist_ok=True)

    # First attempt
    exit_code, log_text, compiler_name, log_path = compile_once(main_tex, build_dir, run_dir)

    report = {
        "status": "ok" if exit_code == 0 else "fail",
        "main_tex": str(main_tex),
        "build_dir": str(build_dir),
        "passes": 1,
        "retry_policy": {"max_retries": int(args.max_retries), "auto_fix_attempted": False},
        "seed": seed,
        "source_map": {"path": str(build_dir / "main.sourcemap.json"), "loaded": False},
        "errors": [],
        "artifacts": {
            "log_file": str(log_path),
            "stdout_file": str(next(run_dir.glob(f"{compiler_name}.stdout.txt"), run_dir / "stdout.txt")),
            "stderr_file": str(next(run_dir.glob(f"{compiler_name}.stderr.txt"), run_dir / "stderr.txt")),
            "pdf_file": str(next(build_dir.glob("*.pdf"), Path(""))) if exit_code == 0 else None,
        },
    }

    if exit_code != 0:
        comp_err, undef_idx = _parse_first_error_and_hints(log_text, project_root)
        source_map = _load_source_map(build_dir)
        report["source_map"]["loaded"] = bool(source_map)
        comp_err, snippet_path, section_path = _resolve_with_source_map(comp_err, source_map, build_dir)

        # If we have an undefined control sequence with no file, try macro-based snippet discovery.
        if comp_err and comp_err.category == "undefined_control_sequence" and not comp_err.file:
            guessed_file, guessed_line, macro = _guess_snippet_from_macro(log_text, undef_idx, build_dir)
            if guessed_file:
                comp_err = replace(comp_err, file=str(guessed_file), line=guessed_line)
                snippet_path = guessed_file
            # (Optional) store macro name in the message for transparency
            if macro and comp_err:
                comp_err = replace(comp_err, message=f"{comp_err.message} (macro {macro})")

        # Attach excerpt if we know a file+line
        if comp_err and comp_err.file and comp_err.line:
            comp_err = replace(
                comp_err,
                code_excerpt=_extract_excerpt(Path(comp_err.file), comp_err.line)
            )

        if comp_err:
            error_entry = {
                "category": comp_err.category,
                "message": comp_err.message,
                "file": comp_err.file,
                "line": comp_err.line,
                "code_excerpt": comp_err.code_excerpt,
            }
            if section_path:
                error_entry["section_file"] = str(section_path)
            report["errors"].append({
                **error_entry,
            })

        # Auto-fix once if enabled
        if args.auto_fix:
            report["retry_policy"]["auto_fix_attempted"] = True
            original_snippet = None
            if snippet_path and snippet_path.exists():
                try:
                    original_snippet = snippet_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    original_snippet = None
            fix_result = invoke_auto_fix(comp_err or CompileError("unknown_error", "unknown", None, None, None),
                                         project_root, build_dir, run_dir, seed,
                                         snippet_path, section_path, log_path, kb_root)
            report["auto_fix"] = fix_result

            if fix_result.get("status") == "fixed":
                agg_result = _rerun_aggregator(plan_path, snippets_dir, build_dir)
                report["auto_fix"]["aggregation"] = agg_result
                source_map = _load_source_map(build_dir)
                report["source_map"]["loaded"] = bool(source_map)
                comp_err, snippet_path, section_path = _resolve_with_source_map(comp_err, source_map, build_dir)

            # Recompile once after fix
            exit_code2, log_text2, compiler_name2, log_path2 = compile_once(main_tex, build_dir, run_dir)
            report["passes"] += 1
            if exit_code2 == 0:
                report["status"] = "ok"
                report["artifacts"]["pdf_file"] = str(next(build_dir.glob("*.pdf"), Path("")))
                if fix_result.get("status") == "fixed":
                    new_snippet = None
                    if snippet_path and snippet_path.exists():
                        try:
                            new_snippet = snippet_path.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            new_snippet = None
                    _log_successful_fix(build_dir, snippet_path, original_snippet, new_snippet,
                                        comp_err or CompileError("unknown_error", "unknown", None, None, None),
                                        fix_result)
            else:
                report["status"] = "fail"
                comp_err2, _ = _parse_first_error_and_hints(log_text2, project_root)
                if comp_err2:
                    report["errors"].append({
                        "category": comp_err2.category,
                        "message": comp_err2.message,
                        "file": comp_err2.file,
                        "line": comp_err2.line,
                        "code_excerpt": comp_err2.code_excerpt,
                    })
                report["artifacts"]["log_file"] = str(log_path2)

    provenance = _infer_provenance(build_dir)
    report["provenance"] = provenance
    text_metrics = _compute_text_metrics_view(build_dir, runs_root, run_id, report.get("status"), provenance)
    report["text_metrics"] = text_metrics
    report["promote_to_kb"] = _should_promote(report, text_metrics)
    report["run_id"] = run_id
    report_path = run_dir / "compile_report.json"
    write_report(report_path, report)
    write_metrics(build_dir, run_id, report_path, report)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
