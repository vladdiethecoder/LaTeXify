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
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

SAFE_PREAMBLE_ALLOWLIST = {
    "amsmath": r"\usepackage{amsmath}",
    "amssymb": r"\usepackage{amssymb}",
    "graphicx": r"\usepackage{graphicx}",
    "xcolor": r"\usepackage{xcolor}",
    "hyperref": r"\usepackage{hyperref}",
}
MAIN_TEX_CANDIDATES = ["main.tex", "paper.tex"]


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


# ------------------------------- Fixers --------------------------------------
def fix_undefined_control_sequence(file_path: Path, line_no: Optional[int]) -> Tuple[bool, str]:
    """
    Minimal safe fix:
    - If line number available: comment out offending line.
    - Else: cautious replacements of deprecated shorthands.
    """
    text = load_text(file_path)
    lines = text.splitlines()
    changed = False
    note = ""

    if line_no is not None and 1 <= line_no <= len(lines):
        i = line_no - 1
        if not lines[i].lstrip().startswith("%"):
            lines[i] = "% [auto-fix] commented unknown command\n" + "% " + lines[i]
            changed = True
            note = f"Commented line {line_no} due to undefined control sequence."
    else:
        repl = {r"\\bf\s+": r"\\textbf{", r"\\it\s+": r"\\textit{"}
        new_text = text
        for pat, sub in repl.items():
            new_text2 = re.sub(pat, sub, new_text)
            if new_text2 != new_text:
                new_text = new_text2
                changed = True
        if changed:
            if not new_text.strip().endswith("}"):
                new_text += "}"
            save_text(file_path, new_text)
            return True, "Replaced deprecated shorthands with textbf/textit."
        note = "No line number; skipped aggressive edits."

    if changed:
        save_text(file_path, "\n".join(lines))
    return changed, note or "No-op."


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


def fix_missing_file_or_package(message: str, build_dir: Path, main_tex: Path) -> Tuple[bool, str, Optional[Path]]:
    msty = re.search(r"File `([^`]+\.sty)` not found", message)
    if msty:
        pkg = Path(msty.group(1)).stem
        changed, note = ensure_preamble_package(main_tex, pkg)
        return changed, note, None

    mfile = re.search(r"I can't find file `([^']+)'\.", message)
    if mfile:
        missing = mfile.group(1)
        base = build_dir / missing
        for ext in (".pdf", ".png", ".jpg"):
            cand = base.with_suffix(ext)
            if cand.exists():
                return False, f"Found {cand.name} in build; update \\includegraphics path.", cand
        return False, f"No alternates found for {missing}.", None

    return False, "No actionable missing file/package pattern.", None


# ------------------------------- Driver --------------------------------------
def _run() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-category", required=True)
    ap.add_argument("--message", required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--file", type=Path)
    ap.add_argument("--line", type=int)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    kb_hint = try_kb_hint(args.error_category, args.message)
    rationale = f"Category={args.error_category}. "
    if kb_hint:
        rationale += f"KB hint: {kb_hint[:180]}..."

    target = args.file if args.file else (detect_main_tex(args.build_dir) or None)
    if not target or not target.exists():
        return {
            "status": "skipped",
            "changed_file": None,
            "rationale": "No target file to edit.",
            "what_changed": "",
            "debug": {"notes": "missing target"},
        }

    status = "skipped"
    what_changed = ""
    changed_file: Optional[Path] = None

    if args.error_category == "undefined_control_sequence":
        ok, note = fix_undefined_control_sequence(target, args.line)
        changed_file = target if ok else None
        what_changed = note
        status = "fixed" if ok else "skipped"

    elif args.error_category in ("env_mismatch", "runaway_argument", "emergency_stop"):
        ok, note = close_unfinished_environment(target)
        changed_file = target if ok else None
        what_changed = note
        status = "fixed" if ok else "skipped"

    elif args.error_category in ("file_not_found", "missing_package"):
        main_tex = detect_main_tex(args.build_dir) or target
        ok, note, _ = fix_missing_file_or_package(args.message, args.build_dir, main_tex)
        changed_file = (main_tex if ok else None)
        what_changed = note
        status = "fixed" if ok else "skipped"

    else:
        status = "skipped"
        what_changed = "No safe rule-based fix for this category."

    return {
        "status": status,
        "changed_file": str(changed_file) if changed_file else None,
        "rationale": rationale,
        "what_changed": what_changed,
        "debug": {"notes": "ok"},
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
