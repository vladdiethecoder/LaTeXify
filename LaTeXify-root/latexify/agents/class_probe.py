# latexify/agents/class_probe.py
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ALLOWED_PACKAGES = ["enumitem", "geometry", "microtype"]  # keep tiny & deterministic
DEFAULT_CLASS = "lix_article"  # we will fall back to scrartcl if missing
FALLBACK_CLASS = "scrartcl"
SEED = 42  # determinism placeholder (not used for randomness here)


@dataclass
class ClassProfile:
    """Description of a document class and the packages it expects."""

    name: str
    fallback: str
    packages: List[str]


def parse_issues(log_text: str) -> List[Dict[str, str]]:
    """Parse latexmk/stdout logs into structured issue dictionaries."""

    issues: List[Dict[str, str]] = []
    for cls_name in re.findall(r"File `([^`]+\.cls)' not found", log_text):
        issues.append({"type": "missing_class", "name": cls_name})
    for pkg_name in re.findall(r"File `([^`]+\.sty)' not found", log_text):
        issues.append({"type": "missing_package", "name": pkg_name})
    if re.search(r"tcrm[0-9]+", log_text) or "TS1/cmr" in log_text:
        issues.append({"type": "missing_ec_fonts", "name": "tcrm/TS1"})
    if re.search(r"mktextfm: .*mf: command not found", log_text):
        issues.append({"type": "missing_metafont", "name": "metafont"})
    return issues


def suggest_fixes(profile: ClassProfile, issues: List[Dict[str, str]]) -> Dict[str, List[str] | List[Dict[str, str]]]:
    """Return installation hints for Fedora/TexLive and fallback notes."""

    fedora: List[str] = []
    tlmgr: List[str] = []
    notes: List[Dict[str, str]] = []

    def _add_unique(target: List[str], value: str) -> None:
        if value not in target:
            target.append(value)

    for issue in issues:
        kind = issue.get("type")
        name = issue.get("name", "")
        if kind == "missing_class":
            notes.append(
                {
                    "action": "fallback_class",
                    "from": profile.name,
                    "to": profile.fallback,
                    "reason": f"{name} not found",
                }
            )
        elif kind == "missing_package":
            pkg = name.replace(".sty", "")
            _add_unique(fedora, f"sudo dnf install texlive-{pkg}")
            _add_unique(tlmgr, f"tlmgr install {pkg}")
        elif kind == "missing_ec_fonts":
            _add_unique(fedora, "sudo dnf install texlive-ec")
            _add_unique(tlmgr, "tlmgr install ec")
        elif kind == "missing_metafont":
            _add_unique(fedora, "sudo dnf install texlive-metafont")
            _add_unique(tlmgr, "tlmgr install metafont")

    return {"fedora": fedora, "tlmgr": tlmgr, "notes": notes}

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log_event(log_path: Path, event: str, **details) -> None:
    """Structured JSONL logging. First arg is *log_path* (not 'path')."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": _utc_now(), "event": event}
    rec.update(details)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def build_probe_tex(doc_class: str, packages: list[str]) -> str:
    pkgs = "\n".join([f"\\usepackage{{{p}}}" for p in packages])
    # Minimal document with a couple of literals that can trigger font lookups.
    return (
        "\\documentclass[11pt]{" + doc_class + "}\n"
        f"{pkgs}\n"
        "\\begin{document}\n"
        "% Probe content with backslash and TS1 glyphs to exercise fonts:\n"
        "Literal backslash: \\textbackslash{} and \\_ underscore.\n"
        "\\begin{itemize}[noitemsep]\n"
        "  \\item Example with numbers 1--2 and math $a_b$.\n"
        "\\end{itemize}\n"
        "\\end{document}\n"
    )

def _run_latexmk(tex_dir: Path, main_name: str = "main.tex") -> tuple[int, str, str]:
    """Run latexmk if present; return (rc, out_tail, err_tail)."""
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", main_name]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(tex_dir),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(SEED)},
        )
        out_tail = p.stdout[-2000:] if p.stdout else ""
        err_tail = p.stderr[-2000:] if p.stderr else ""
        return p.returncode, out_tail, err_tail
    except FileNotFoundError:
        # latexmk not installed — report gracefully
        return 127, "", "latexmk_not_found"

def diagnose_from_logs(out_tail: str, err_tail: str, tex: str) -> dict:
    issues = []

    # Missing class
    m = re.search(r"File `([^`]+\.cls)' not found", out_tail) or \
        re.search(r"File `([^`]+\.cls)' not found", err_tail)
    if m:
        issues.append({"type": "missing_class", "name": m.group(1)})

    # Missing package .sty
    for sty in ["microtype.sty"]:
        if (sty in out_tail) or (sty in err_tail):
            issues.append({"type": "missing_sty", "name": sty})

    # TS1 font (EC) missing -> looks like tcrm1095 in your logs
    if "tcrm1095" in out_tail or "tcrm1095" in err_tail:
        issues.append({"type": "missing_font_metrics", "name": "TS1/EC (tcrm1095)"})

    return {"issues": issues}

def main():
    ap = argparse.ArgumentParser(description="Probe LaTeX class availability and minimal compile.")
    ap.add_argument("--class", dest="doc_class", default=DEFAULT_CLASS)
    ap.add_argument("--out_root", default="build/class_probe")
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.doc_class
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "probe.log.jsonl"

    log_event(log_path, "probe_begin", class_name=args.doc_class, out=str(out_dir))

    tex = build_probe_tex(args.doc_class, ALLOWED_PACKAGES)
    tex_path = out_dir / "main.tex"
    tex_path.write_text(tex, encoding="utf-8")
    log_event(log_path, "write_main", file=str(tex_path), bytes=len(tex))

    report = {
        "ok": False,
        "class": args.doc_class,
        "fallback": FALLBACK_CLASS,
        "packages": ALLOWED_PACKAGES,
        "issues": [],
        "suggestions": {
            "fedora": [],
            "tlmgr": [],
            "notes": [],
        },
        "outputs": {
            "tex": str(tex_path),
            "log": str(log_path),
            "pdf": str(out_dir / "main.pdf"),
        },
    }

    if args.no_compile:
        report["ok"] = True
        (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log_event(log_path, "probe_end", class_name=args.doc_class, ok=True, issues=0)
        print(json.dumps({"ok": True, "report": str(out_dir / "probe_report.json")}))
        return

    rc, out_tail, err_tail = _run_latexmk(out_dir)
    log_event(log_path, "compile_try", rc=rc, out_tail=out_tail, err_tail=err_tail)

    diag = diagnose_from_logs(out_tail, err_tail, tex)
    report["issues"] = diag["issues"]

    # Suggestions (Fedora), kept short & sourced in docs
    missing_class = any(i["type"] == "missing_class" for i in diag["issues"])
    if missing_class and args.doc_class == "lix_article":
        report["suggestions"]["notes"].append({
            "action": "fallback_class",
            "to": FALLBACK_CLASS,
            "reason": "lix_article.cls not found"
        })
    if "latexmk_not_found" in err_tail:
        report["suggestions"]["fedora"].append("sudo dnf install latexmk")

    if any(i.get("name") == "microtype.sty" for i in diag["issues"]):
        report["suggestions"]["fedora"].append("sudo dnf install texlive-microtype")

    if any(i.get("name") == "TS1/EC (tcrm1095)" for i in diag["issues"]):
        # EC fonts + metafont are safe bets for the tcrm1095 class of errors
        report["suggestions"]["fedora"].extend([
            "sudo dnf install texlive-ec",
            "sudo dnf install texlive-metafont",
        ])

    report["ok"] = (rc == 0)
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_event(log_path, "probe_end", class_name=args.doc_class, ok=report["ok"], issues=len(report["issues"]))

    print(json.dumps({"ok": report["ok"], "report": str(out_dir / "probe_report.json")}))
    return

if __name__ == "__main__":
    main()
