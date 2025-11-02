#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_fix.py — Self-Correction Agent (single-pass) for LaTeXify.

Hard guarantees:
- Emits exactly ONE JSON object to *stdout FD* and exits, even on unexpected errors.
- Suppresses any incidental prints from optional KB imports so stdout stays clean.

Output JSON (stdout):
{
  "status": "fixed" | "skipped" | "error",
  "changed_file": "<path or null>",
  "rationale": "<short text>",
  "what_changed": "<short diff-like note>",
  "debug": { "notes": "...", "exception": "<type: msg>"? }
}
"""

from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

SAFE_PREAMBLE_ALLOWLIST = {
    "amsmath": r"\usepackage{amsmath}",
    "amssymb": r"\usepackage{amssymb}",
    "booktabs": r"\usepackage{booktabs}",
    "caption": r"\usepackage{caption}",
    "cleveref": r"\usepackage{cleveref}",
    "graphicx": r"\usepackage{graphicx}",
    "hyperref": r"\usepackage{hyperref}",
    "mathtools": r"\usepackage{mathtools}",
    "physics": r"\usepackage{physics}",
    "siunitx": r"\usepackage{siunitx}",
    "xcolor": r"\usepackage{xcolor}",
    "bm": r"\usepackage{bm}",
    "cancel": r"\usepackage{cancel}",
}
MAIN_TEX_CANDIDATES = ["main.tex", "paper.tex"]

PACKAGE_TO_CAPABILITY = {
    "amsmath": "amsmath",
    "amssymb": "amssymb",
    "mathtools": "mathtools",
    "graphicx": "graphicx",
    "booktabs": "booktabs",
    "siunitx": "siunitx",
    "xcolor": "xcolor",
    "hyperref": "hyperref",
    "caption": "caption",
    "cleveref": "cleveref",
    "bm": "bm",
    "physics": "physics",
    "cancel": "cancel",
}

MACRO_CAPABILITY_HINTS = {
    "\\mathbb": "amssymb",
    "\\mathcal": "amssymb",
    "\\mathfrak": "amssymb",
    "\\bm": "bm",
    "\\boldsymbol": "amsmath",
    "\\qty": "physics",
    "\\dv": "physics",
    "\\pdv": "physics",
    "\\abs": "physics",
    "\\norm": "physics",
    "\\toprule": "booktabs",
    "\\midrule": "booktabs",
    "\\bottomrule": "booktabs",
    "\\includegraphics": "graphicx",
    "\\SI": "siunitx",
    "\\si": "siunitx",
    "\\cancel": "cancel",
    "\\textcolor": "xcolor",
    "\\autoref": "hyperref",
    "\\cref": "cleveref",
    "\\Cref": "cleveref",
}


# ----------- Robust emitter: write JSON to FD 1 and hard-exit ----------------
def _emit_json_and_exit(payload: dict, exit_code: int = 0) -> None:
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="replace")
        # write to stdout FD directly so pytest FD-capture and subprocess pipes see it
        os.write(1, data)
        os.write(1, b"\n")
    finally:
        # bypass Python shutdown/buffering entirely
        os._exit(exit_code)


# ----------------------------- I/O helpers -----------------------------------
def load_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def save_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def detect_main_tex(build_dir: Path) -> Optional[Path]:
    for name in MAIN_TEX_CANDIDATES:
        cand = build_dir / name
        if cand.exists():
            return cand
    files = list(build_dir.glob("*.tex"))
    return files[0] if files else None

# --------------------------- Capability helpers ------------------------------

def capabilities_from_context(context: str) -> Set[str]:
    caps: Set[str] = set()
    if not context:
        return caps
    for pkg, cap in PACKAGE_TO_CAPABILITY.items():
        if f"\\usepackage{{{pkg}}}" in context:
            caps.add(cap)
    return caps


def update_meta_capabilities(snippet_path: Path, new_caps: Set[str]) -> Tuple[bool, List[str]]:
    if not new_caps:
        return False, []
    meta_path = snippet_path.with_suffix(".meta.json")
    data: Dict = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    caps = set(data.get("capabilities", []) or [])
    before = sorted(caps)
    caps.update(new_caps)
    after = sorted(caps)
    if after != before:
        data["capabilities"] = after
        meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, after
    return False, after


def extract_macro_from_message(message: str) -> Optional[str]:
    if not message:
        return None
    match = re.search(r"(\\[A-Za-z@]+)", message)
    return match.group(1) if match else None


def macro_from_snippet_line(text: str, line_no: Optional[int]) -> Optional[str]:
    if not text or not line_no:
        return None
    lines = text.splitlines()
    if not (1 <= line_no <= len(lines)):
        return None
    candidate_line = lines[line_no - 1]
    match = re.search(r"(\\[A-Za-z@]+)", candidate_line)
    return match.group(1) if match else None


MACRO_FALLBACK_DEFINITIONS = {
    "\\toprule": {
        "marker": "\\providecommand{\\toprule}",
        "lines": [
            "% [auto-fix] fallback definition for \\toprule",
            r"\\providecommand{\\toprule}{\\hline}",
        ],
    },
    "\\midrule": {
        "marker": "\\providecommand{\\midrule}",
        "lines": [
            "% [auto-fix] fallback definition for \\midrule",
            r"\\providecommand{\\midrule}{\\hline}",
        ],
    },
    "\\bottomrule": {
        "marker": "\\providecommand{\\bottomrule}",
        "lines": [
            "% [auto-fix] fallback definition for \\bottomrule",
            r"\\providecommand{\\bottomrule}{\\hline}",
        ],
    },
    "\\bm": {
        "marker": "\\providecommand{\\bm}",
        "lines": [
            "% [auto-fix] fallback definition for \\bm",
            r"\\providecommand{\\bm}[1]{\\mathbf{#1}}",
        ],
    },
    "\\autoref": {
        "marker": "\\providecommand{\\autoref}",
        "lines": [
            "% [auto-fix] fallback definition for \\autoref",
            r"\\providecommand{\\autoref}[1]{\\ref{#1}}",
        ],
    },
    "\\cref": {
        "marker": "\\providecommand{\\cref}",
        "lines": [
            "% [auto-fix] fallback definition for \\cref",
            r"\\providecommand{\\cref}[1]{\\ref{#1}}",
        ],
    },
    "\\Cref": {
        "marker": "\\providecommand{\\Cref}",
        "lines": [
            "% [auto-fix] fallback definition for \\Cref",
            r"\\providecommand{\\Cref}[1]{\\ref{#1}}",
        ],
    },
    "\\todo": {
        "marker": "\\providecommand{\\todo}",
        "lines": [
            "% [auto-fix] fallback definition for \\todo",
            r"\\providecommand{\\todo}[1]{\\textbf{TODO: }#1}",
        ],
    },
}


def ensure_fallback_definition(
    snippet_path: Path,
    text: str,
    macro: Optional[str],
) -> Tuple[bool, str, str]:
    if not macro:
        return False, text, "No macro identified for fallback."
    fallback = MACRO_FALLBACK_DEFINITIONS.get(macro)
    if not fallback:
        return False, text, f"No fallback available for {macro}."
    marker = fallback["marker"]
    if marker in text:
        return False, text, "Fallback already present."
    lines = text.splitlines()
    insert_at = 0
    while insert_at < len(lines) and lines[insert_at].strip().startswith("%"):
        insert_at += 1
    insertion = list(fallback["lines"])
    new_lines = lines[:insert_at] + insertion + lines[insert_at:]
    if insertion and insert_at + len(insertion) < len(new_lines):
        if new_lines[insert_at + len(insertion)].strip():
            new_lines.insert(insert_at + len(insertion), "")
    new_text = "\n".join(new_lines)
    if not new_text.endswith("\n"):
        new_text += "\n"
    save_text(snippet_path, new_text)
    return True, new_text, f"Inserted fallback definition for {macro}."


def retrieve_kb_context(kb_root: Path, query: str, limit: int = 3) -> str:
    docs_path = kb_root / "latex_docs.jsonl"
    if not docs_path.exists():
        return ""
    tokens = [tok for tok in re.findall(r"[A-Za-z@]+", query.lower()) if tok]
    scored: List[Tuple[int, str]] = []
    try:
        with docs_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                body_parts = [
                    rec.get("title") or "",
                    rec.get("question") or "",
                    rec.get("answer") or "",
                    "\n".join(rec.get("code_blocks") or []),
                ]
                body = "\n".join(part for part in body_parts if part)
                text_lower = body.lower()
                score = sum(text_lower.count(tok) for tok in tokens)
                if score > 0:
                    scored.append((score, body))
    except Exception:
        return ""
    scored.sort(key=lambda t: -t[0])
    return "\n\n".join(body for _, body in scored[:limit])


def macro_capabilities(macro: Optional[str]) -> Set[str]:
    if not macro:
        return set()
    cap = MACRO_CAPABILITY_HINTS.get(macro)
    return {cap} if cap else set()



# --------------------------- Optional KB lookup ------------------------------
def try_kb_hint(error_category: str, message: str) -> Optional[str]:
    """
    Consult scripts/retrieval_bundle.py: latex_kb_hints(query) -> str (if present).
    Silence stdout/stderr during import/call so our stdout remains pure JSON only.
    """
    try:
        import importlib.util
        mod_path = HERE / "retrieval_bundle.py"
        if not mod_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("retrieval_bundle", str(mod_path))
        rb = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert spec.loader is not None
            spec.loader.exec_module(rb)  # type: ignore
        if hasattr(rb, "latex_kb_hints"):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return rb.latex_kb_hints(f"{error_category}: {message}")  # type: ignore
    except Exception:
        return None
    return None


def gather_capability_hints(
    message: str,
    snippet_text: str,
    line_no: Optional[int],
    kb_context: str,
    primary_macro: Optional[str],
) -> Set[str]:
    caps = set()
    macro = primary_macro or extract_macro_from_message(message)
    caps.update(macro_capabilities(macro))
    caps.update(capabilities_from_context(kb_context))
    if not caps and snippet_text:
        extra_macro = macro_from_snippet_line(snippet_text, line_no)
        if extra_macro:
            caps.update(macro_capabilities(extra_macro))
    return caps


# ------------------------------- Fixers --------------------------------------
def fix_undefined_control_sequence(
    file_path: Path,
    line_no: Optional[int],
    message: str,
    kb_context: str,
) -> Tuple[bool, str, List[str]]:
    text = load_text(file_path)
    macro = extract_macro_from_message(message) or macro_from_snippet_line(text, line_no)
    fallback_changed, text, fallback_note = ensure_fallback_definition(file_path, text, macro)
    caps = gather_capability_hints(message, text, line_no, kb_context, macro)
    meta_changed, updated_caps = update_meta_capabilities(file_path, caps)
    notes: List[str] = []
    if meta_changed:
        notes.append(f"meta capabilities -> {updated_caps}")
    if fallback_changed:
        notes.append(fallback_note)
    changed = meta_changed or fallback_changed
    return changed, "; ".join(notes) if notes else "No capability change.", updated_caps


def ensure_preamble_package(main_tex: Path, pkg: str) -> Tuple[bool, str]:
    if pkg not in SAFE_PREAMBLE_ALLOWLIST:
        return False, f"Package {pkg} not on allowlist."
    text = load_text(main_tex)
    if f"\\usepackage{{{pkg}}}" in text:
        return False, f"{pkg} already present."
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("\\documentclass"):
            lines.insert(i + 1, SAFE_PREAMBLE_ALLOWLIST[pkg])
            save_text(main_tex, "\n".join(lines))
            return True, f"Inserted \\usepackage{{{pkg}}}."
    save_text(main_tex, SAFE_PREAMBLE_ALLOWLIST[pkg] + "\n" + text)
    return True, f"Prepended \\usepackage{{{pkg}}}."


def close_unfinished_environment(file_path: Path) -> Tuple[bool, str]:
    text = load_text(file_path)
    begins = re.findall(r"\\begin\{([^\}]+)\}", text)
    ends = re.findall(r"\\end\{([^\}]+)\}", text)
    for env in begins:
        if begins.count(env) > ends.count(env):
            text += f"\n% [auto-fix] closing unbalanced environment\n\\end{{{env}}}\n"
            save_text(file_path, text)
            return True, f"Appended \\end{{{env}}}."
    return False, "No unbalanced environment detected."


_MISSING_FILE_PATTERNS = [
    re.compile(r"LaTeX Error: File `([^`]+)` not found\."),
    re.compile(r"! I can't find file `([^']+)'\."),
]


def _extract_missing_filename(message: str) -> Optional[str]:
    if not message:
        return None
    for pattern in _MISSING_FILE_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1)
    return None


def _locate_asset_on_disk(missing: str, build_dir: Path) -> Optional[Path]:
    candidates = []
    target = Path(missing)
    if target.is_absolute():
        candidates.append(target)
    candidates.append(build_dir / missing)
    if target.name != missing:
        candidates.append(build_dir / target.name)
    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except Exception:
            continue
    return None


def _replace_includegraphics_with_placeholder(snippet_path: Path, missing: str) -> Tuple[bool, str]:
    if not snippet_path.exists():
        return False, "Snippet not found."
    text = load_text(snippet_path)
    if not text:
        return False, "Snippet empty; nothing to replace."
    tokens = {missing, Path(missing).as_posix(), Path(missing).name}
    lines = text.splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        if "\\includegraphics" not in line:
            continue
        if not any(token in line for token in tokens):
            continue
        indent = line[: len(line) - len(line.lstrip())]
        placeholder = f"{indent}\\fbox{{Missing Asset: {Path(missing).name}}} % [auto-fix missing asset]"
        lines[idx] = placeholder
        replaced = True
        break
    if not replaced:
        return False, "Includegraphics reference not found in snippet."
    new_text = "\n".join(lines)
    if not new_text.endswith("\n"):
        new_text += "\n"
    save_text(snippet_path, new_text)
    return True, f"Replaced includegraphics with placeholder for {Path(missing).name}."


def fix_missing_file_or_package(message: str, build_dir: Path, main_tex: Path, snippet_path: Optional[Path] = None) -> Tuple[bool, str, Optional[Path]]:
    msty = re.search(r"File `([^`]+\.sty)` not found", message)
    if msty:
        pkg = Path(msty.group(1)).stem
        changed, note = ensure_preamble_package(main_tex, pkg)
        return changed, note, None

    missing = _extract_missing_filename(message)
    if missing:
        located = _locate_asset_on_disk(missing, build_dir)
        if located:
            return False, f"File {missing} exists at {located}; check path usage.", located
        if snippet_path:
            changed, note = _replace_includegraphics_with_placeholder(snippet_path, missing)
            if changed:
                return True, note, snippet_path
        base = build_dir / missing
        for ext in (".pdf", ".png", ".jpg"):
            cand = base.with_suffix(ext)
            if cand.exists():
                return False, f"Found {cand.name} in build; update \\includegraphics path.", cand
        return False, f"Missing file {missing}; no snippet updated.", None

    return False, "No actionable missing file/package pattern.", None


# ------------------------------- Driver --------------------------------------
def _run() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument('--error-category', required=True)
    ap.add_argument('--message', required=True)
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--build-dir', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--file', type=Path)
    ap.add_argument('--line', type=int)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--snippet', type=Path, help='Direct path to snippet that should be rewritten')
    ap.add_argument('--section', type=Path, help='Section file associated with the error (informational)')
    ap.add_argument('--log-path', type=Path, help='Path to the LaTeX compiler log')
    ap.add_argument('--kb-root', type=Path, default=PROJECT_ROOT / 'kb' / 'latex')
    args = ap.parse_args()

    kb_hint = try_kb_hint(args.error_category, args.message)
    rationale = f"Category={args.error_category}. "
    if kb_hint:
        rationale += f"KB hint: {kb_hint[:180]}..."

    snippet = args.snippet or args.file
    if snippet is None:
        snippet = detect_main_tex(args.build_dir)

    kb_context = retrieve_kb_context(args.kb_root, args.message)
    if kb_context:
        rationale += ' Retrieved KB context.'

    target = snippet if snippet else None
    if not target or not target.exists():
        return {
            'status': 'skipped',
            'changed_file': None,
            'rationale': 'No target file to edit.',
            'what_changed': '',
            'capabilities': [],
            'kb_context': kb_context,
            'debug': {'notes': 'missing target'},
        }

    status = 'skipped'
    what_changed = ''
    changed_file: Optional[Path] = None
    capabilities: List[str] = []

    if args.error_category == 'undefined_control_sequence':
        ok, note, caps = fix_undefined_control_sequence(target, args.line, args.message, kb_context)
        changed_file = target if ok else None
        what_changed = note
        capabilities = caps
        status = 'fixed' if ok else 'skipped'

    elif args.error_category in ('env_mismatch', 'runaway_argument', 'emergency_stop'):
        ok, note = close_unfinished_environment(target)
        changed_file = target if ok else None
        what_changed = note
        status = 'fixed' if ok else 'skipped'

    elif args.error_category in ('file_not_found', 'missing_package'):
        main_tex = detect_main_tex(args.build_dir) or target
        ok, note, changed_path = fix_missing_file_or_package(args.message, args.build_dir, main_tex, target)
        if ok:
            changed_file = changed_path if changed_path else main_tex
        else:
            changed_file = changed_path
        what_changed = note
        status = 'fixed' if ok else 'skipped'

    else:
        status = 'skipped'
        what_changed = 'No safe rule-based fix for this category.'

    return {
        'status': status,
        'changed_file': str(changed_file) if changed_file else None,
        'rationale': rationale,
        'what_changed': what_changed,
        'capabilities': capabilities,
        'kb_context': kb_context,
        'debug': {'notes': 'ok', 'section': str(args.section) if args.section else None},
    }


def main() -> None:
    try:
        payload = _run()
        # exit_code = 0 for fixed/skipped; 1 only for "error"
        exit_code = 0 if payload.get("status") in ("fixed", "skipped") else 1
        _emit_json_and_exit(payload, exit_code)
    except Exception as e:
        _emit_json_and_exit({
            "status": "error",
            "changed_file": None,
            "rationale": "Unhandled exception in auto_fix",
            "what_changed": "",
            "debug": {"notes": f"{type(e).__name__}: {e}"},
        }, exit_code=1)


if __name__ == "__main__":
    main()
