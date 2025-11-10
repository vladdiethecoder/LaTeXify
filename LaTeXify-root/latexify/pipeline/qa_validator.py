from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .critic_agent import CriticAgent
from .specialist_router import SpecialistDecision
from .synth_shared import SpecialistPrompt
from .tex_assembler import determine_packages, load_plan, parse_tasks

PACKAGE_RULES: List[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\\chemfig"), "\\usepackage{chemfig}", "\\chemfig detected without chemfig package"),
    (re.compile(r"\\ce\{"), "\\usepackage[version=4]{mhchem}", "mhchem macros detected"),
    (re.compile(r"\\SI\{"), "\\usepackage{siunitx}", "siunitx unit macros detected"),
    (re.compile(r"\\begin\{tikzpicture"), "\\usepackage{tikz}", "tikzpicture environment requires tikz"),
]
TODO_RX = re.compile(r"TODO|\\todo|\?\?\?", re.IGNORECASE)
ENV_RX = re.compile(r"\\(begin|end)\{([A-Za-z*@0-9_-]+)\}")
PACKAGE_LINE_RX = re.compile(r"\\usepackage(?:\[[^\]]+\])?\{([^}]+)\}")
CHKTEX_LINE_RX = re.compile(r"^(Warning|Error)\s+(\d+)\s+in\s+.+?line\s+(\d+):\s*(.+)$")
UNDEFINED_REF_RX = re.compile(r"Reference `([^']+)'\s+on page", re.IGNORECASE)
MISSING_PKG_RX = re.compile(r"File `([^']+)\.(?:sty|cls)' not found", re.IGNORECASE)
MATH_ERROR_RX = re.compile(
    r"Missing \$ inserted|Math mode|Bad math environment delimiter|Extra alignment tab", re.IGNORECASE
)
CHK_TEX_BIN = shutil.which("chktex")


@dataclass
class CompileSummary:
    attempted: bool
    ok: bool
    engine: str
    log_path: Optional[Path]
    note: Optional[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_preflight(
    plan_path: Path,
    snippets_dir: Path,
    *,
    build_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    attempt_compile: bool = False,
    max_passes: int = 1,
    main_tex: Optional[Path] = None,
) -> Path:
    plan_schema = load_plan(plan_path)
    qa_root = report_dir or (build_dir or plan_path.parent) / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    snippets_dir = snippets_dir.resolve()
    snippets: Dict[str, str] = {}
    for path in sorted(snippets_dir.glob("*.tex")):
        snippets[path.stem] = path.read_text(encoding="utf-8")
    preamble_text = snippets.get("PREAMBLE", "")
    existing_packages = _extract_packages(preamble_text)
    inferred_packages = determine_packages(plan_schema.content_flags, snippets.values())
    known_packages = _dedupe(existing_packages + inferred_packages)
    findings: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    auto_fixes: List[Dict[str, Any]] = []
    fixes_dir = qa_root / "fixes"
    fixes_dir.mkdir(parents=True, exist_ok=True)
    snippet_package_requests: Dict[str, List[str]] = {}

    for snippet_id, text in snippets.items():
        if snippet_id in {"PREAMBLE", "TITLE"}:
            continue
        meta = _load_meta(snippets_dir / f"{snippet_id}.tex")
        if meta.get("auto_flagged"):
            findings.append(
                {
                    "snippet": snippet_id,
                    "severity": "warning",
                    "code": "auto_flagged",
                    "detail": "Judge marked this chunk as flagged; review before compile.",
                }
            )
        if TODO_RX.search(text):
            findings.append(
                {
                    "snippet": snippet_id,
                    "severity": "info",
                    "code": "todo_marker",
                    "detail": "Placeholder TODO markers detected.",
                }
            )
        missing_pkgs = _detect_missing_packages(text, known_packages)
        suggestions.extend(missing_pkgs)
        if missing_pkgs:
            snippet_package_requests.setdefault(snippet_id, []).extend(missing_pkgs)
        env_issues = _detect_env_issues(text)
        if env_issues:
            findings.append(
                {
                    "snippet": snippet_id,
                    "severity": "warning",
                    "code": "environment_imbalance",
                    "detail": f"Unbalanced environments: {[issue['env'] for issue in env_issues]}",
                }
            )
            fixed = _auto_fix_environments(text, env_issues, max_passes)
            if fixed != text:
                fix_path = fixes_dir / f"{snippet_id}.fixed.tex"
                fix_path.write_text(fixed, encoding="utf-8")
                auto_fixes.append({"snippet": snippet_id, "path": str(fix_path)})
        chktex_findings = _run_chktex(snippets_dir / f"{snippet_id}.tex", snippet_id)
        findings.extend(chktex_findings)

    suggestions = _dedupe(suggestions)
    preamble_path = snippets_dir / "PREAMBLE.tex"
    suggestions = _apply_package_suggestions(
        preamble_path,
        suggestions,
        snippet_package_requests,
        snippets_dir,
        fixes_dir,
        auto_fixes,
    )

    compile_summary = CompileSummary(False, False, "pdflatex", None, "compile skipped")
    compile_log_excerpt = ""
    if attempt_compile:
        main_candidate = main_tex or _synthesize_main(plan_schema, snippets_dir, qa_root)
        compile_summary, compile_log_excerpt = _attempt_compile(main_candidate, qa_root)
        if compile_log_excerpt:
            compile_findings, compile_missing = _parse_compile_findings(compile_log_excerpt)
            findings.extend(compile_findings)
            if compile_missing:
                remaining_compile = _apply_package_suggestions(
                    preamble_path,
                    compile_missing,
                    {},
                    snippets_dir,
                    fixes_dir,
                    auto_fixes,
                )
                suggestions = _dedupe(suggestions + remaining_compile)
        if attempt_compile and not compile_summary.ok:
            critic_finding = _run_critic_recheck(plan_schema, snippets_dir, qa_root)
            if critic_finding:
                findings.append(critic_finding)

    report = {
        "created_at": _now_iso(),
        "plan": str(plan_path),
        "snippets_checked": len(snippets) - (1 if "PREAMBLE" in snippets else 0),
        "qa_dir": str(qa_root),
        "suggested_packages": suggestions,
        "findings": findings,
        "auto_fixes": auto_fixes,
        "compile": {
            "attempted": compile_summary.attempted,
            "ok": compile_summary.ok,
            "engine": compile_summary.engine,
            "log_path": str(compile_summary.log_path) if compile_summary.log_path else None,
            "note": compile_summary.note,
        },
    }
    report_path = qa_root / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _extract_packages(preamble: str) -> List[str]:
    packages: List[str] = []
    for line in preamble.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\usepackage"):
            packages.append(stripped)
    return packages


def _load_meta(snippet_path: Path) -> Dict[str, Any]:
    meta_path = snippet_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _detect_missing_packages(text: str, known_packages: Sequence[str]) -> List[str]:
    suggestions: List[str] = []
    for pattern, pkg_line, reason in PACKAGE_RULES:
        if pattern.search(text) and pkg_line not in known_packages and pkg_line not in suggestions:
            suggestions.append(pkg_line)
    return suggestions


def _detect_env_issues(text: str) -> List[Dict[str, Any]]:
    stack: List[tuple[str, int]] = []
    issues: List[Dict[str, Any]] = []
    for match in ENV_RX.finditer(text):
        kind, env = match.group(1), match.group(2)
        if kind == "begin":
            stack.append((env, match.start()))
        else:
            if stack and stack[-1][0] == env:
                stack.pop()
            else:
                issues.append({"code": "orphan_end", "env": env, "pos": match.start()})
    for env, pos in stack:
        issues.append({"code": "missing_end", "env": env, "pos": pos})
    return issues


def _auto_fix_environments(text: str, issues: List[Dict[str, Any]], max_passes: int) -> str:
    fixed = text
    closes_added: Dict[str, int] = {}
    for issue in issues:
        env = issue["env"]
        if issue["code"] == "missing_end":
            count = closes_added.get(env, 0)
            if count >= max_passes:
                continue
            fixed = fixed.rstrip() + f"\n\\end{{{env}}}\n"
            closes_added[env] = count + 1
        elif issue["code"] == "orphan_end":
            pattern = re.compile(rf"\\end\{{{re.escape(env)}\}}", re.MULTILINE)
            fixed, replaced = pattern.subn(f"% QA removed unmatched \\end{{{env}}}", fixed, count=1)
            if replaced == 0:
                continue
    return fixed


def _extract_package_name(pkg_line: str) -> Optional[str]:
    match = PACKAGE_LINE_RX.search(pkg_line)
    if not match:
        return None
    name = match.group(1).strip()
    if not name:
        return None
    return name


def _ensure_preamble_packages(preamble_path: Path, packages: Sequence[str]) -> List[str]:
    if not packages:
        return []
    inserted: List[str] = []
    if preamble_path.exists():
        preamble = preamble_path.read_text(encoding="utf-8")
    else:
        preamble = "\\documentclass{article}\n\\begin{document}\n"
    lines = preamble.splitlines()
    try:
        doc_idx = next(idx for idx, line in enumerate(lines) if "\\begin{document}" in line)
    except StopIteration:
        doc_idx = len(lines)
    for pkg in packages:
        if not pkg or pkg in preamble:
            continue
        lines.insert(doc_idx, pkg)
        inserted.append(pkg)
        doc_idx += 1
    if inserted:
        preamble_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inserted


def _comment_snippet_packages(
    snippet_path: Path,
    packages: Sequence[str],
) -> Tuple[bool, str]:
    if not packages or not snippet_path.exists():
        return False, ""
    text = snippet_path.read_text(encoding="utf-8")
    modified = text
    changed = False
    noted_pkgs: List[str] = []
    for pkg_line in packages:
        pkg_name = _extract_package_name(pkg_line)
        if not pkg_name:
            continue
        pattern = re.compile(rf"(?m)^(\s*\\usepackage[^\n]*\{{\s*{re.escape(pkg_name)}\s*\}}[^\n]*?)$")
        if not pattern.search(modified):
            continue
        modified = pattern.sub(r"% QA commented package: \1", modified, count=1)
        changed = True
        noted_pkgs.append(pkg_name)
    if changed:
        snippet_path.write_text(modified, encoding="utf-8")
        return True, ", ".join(noted_pkgs)
    return False, ""


def _apply_package_suggestions(
    preamble_path: Path,
    packages: Sequence[str],
    snippet_pkg_map: Dict[str, List[str]],
    snippets_dir: Path,
    fixes_dir: Path,
    auto_fixes: List[Dict[str, Any]],
) -> List[str]:
    if not packages:
        return []
    inserted = _ensure_preamble_packages(preamble_path, packages)
    if inserted:
        auto_fixes.append(
            {
                "snippet": "PREAMBLE",
                "path": str(preamble_path),
                "detail": f"Inserted packages: {', '.join(inserted)}",
            }
        )
    for snippet_id, pkg_lines in snippet_pkg_map.items():
        snippet_path = snippets_dir / f"{snippet_id}.tex"
        changed, noted = _comment_snippet_packages(snippet_path, pkg_lines)
        if changed:
            auto_fixes.append(
                {
                    "snippet": snippet_id,
                    "path": str(snippet_path),
                    "detail": f"Commented inline packages: {noted}",
                }
            )
    remaining = [pkg for pkg in packages if pkg not in inserted]
    return remaining


def _run_chktex(snippet_path: Path, snippet_id: str) -> List[Dict[str, Any]]:
    if not CHK_TEX_BIN or not snippet_path.exists():
        return []
    try:
        proc = subprocess.run(
            [CHK_TEX_BIN, "-q", str(snippet_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    findings: List[Dict[str, Any]] = []
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in output.splitlines():
        match = CHKTEX_LINE_RX.match(line.strip())
        if not match:
            continue
        level, code, line_no, message = match.groups()
        severity = "error" if level.lower() == "error" else "warning"
        issue_code = "chktex"
        if "Missing $" in message or "Math" in message:
            issue_code = "math_delimiter"
        elif "undefined" in message.lower():
            issue_code = "undefined_reference"
        findings.append(
            {
                "snippet": snippet_id,
                "severity": severity,
                "code": issue_code,
                "detail": f"chktex {level} {code} (line {line_no}): {message}",
            }
        )
    return findings


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _parse_compile_findings(log: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    findings: List[Dict[str, Any]] = []
    missing_packages: List[str] = []
    for match in MISSING_PKG_RX.finditer(log):
        pkg = match.group(1).strip()
        if not pkg:
            continue
        package_line = f"\\usepackage{{{pkg}}}"
        missing_packages.append(package_line)
        findings.append(
            {
                "snippet": "PREAMBLE",
                "severity": "error",
                "code": "missing_package",
                "detail": f"Package '{pkg}' missing during compile; queued for insertion.",
            }
        )
    for match in UNDEFINED_REF_RX.finditer(log):
        ref = match.group(1)
        findings.append(
            {
                "snippet": None,
                "severity": "warning",
                "code": "undefined_reference",
                "detail": f"Reference '{ref}' reported as undefined.",
            }
        )
    if MATH_ERROR_RX.search(log):
        findings.append(
            {
                "snippet": None,
                "severity": "error",
                "code": "math_delimiter",
                "detail": "Compile log reported mismatched math delimiters.",
            }
        )
    return findings, missing_packages


def _critic_decision() -> SpecialistDecision:
    prompt = SpecialistPrompt(version="qa-validator", body="QA validator critic rerun")
    return SpecialistDecision(
        name="qa_validator",
        handler=lambda bundle: ("", []),
        reason="qa_validator_retry",
        prompt=prompt,
        metadata={},
    )


def _run_critic_recheck(plan_schema, snippets_dir: Path, qa_root: Path) -> Optional[Dict[str, Any]]:
    try:
        critic = CriticAgent(plan_schema.dict())
    except Exception:
        return None
    decision = _critic_decision()
    tasks = parse_tasks(plan_schema)
    for task in tasks:
        if task.task_id in {"PREAMBLE", "TITLE"}:
            continue
        snippet_path = snippets_dir / f"{task.task_id}.tex"
        if not snippet_path.exists():
            continue
        snippet_text = snippet_path.read_text(encoding="utf-8")
        result = critic.review(
            snippet_text,
            bundle={"task_id": task.task_id, "title": task.title},
            decision=decision,
            attempt=1,
            feedback_history=[],
        )
        if not result.accepted:
            log_path = qa_root / f"critic_{task.task_id}.txt"
            log_path.write_text(result.feedback, encoding="utf-8")
            return {
                "snippet": task.task_id,
                "severity": "error",
                "code": "critic_feedback",
                "detail": result.feedback,
            }
    return None


def _synthesize_main(plan: Dict[str, Any], snippets_dir: Path, qa_root: Path) -> Path:
    tasks = parse_tasks(plan)
    preamble_path = snippets_dir / "PREAMBLE.tex"
    body_parts: List[str] = []
    for task in tasks:
        snippet_path = snippets_dir / f"{task.task_id}.tex"
        if snippet_path.exists():
            body_parts.append(snippet_path.read_text(encoding="utf-8"))
    body = "\n".join(body_parts).strip()
    preamble = preamble_path.read_text(encoding="utf-8") if preamble_path.exists() else "\\documentclass{article}\n\\begin{document}\n"
    if "\\begin{document}" not in preamble:
        preamble += "\n\\begin{document}\n"
    if not body:
        body = "% No body content"
    if "\\end{document}" not in body:
        body += "\n\\end{document}\n"
    synthetic = qa_root / "preflight_main.tex"
    synthetic.write_text(preamble + "\n" + body, encoding="utf-8")
    return synthetic


def _attempt_compile(main_tex: Path, qa_root: Path) -> tuple[CompileSummary, str]:
    latexmk = shutil.which("latexmk")
    engine = "pdflatex"
    log_path = qa_root / "preflight-compile.log"
    if not latexmk:
        note = "latexmk not found; static analysis only"
        log_path.write_text(note, encoding="utf-8")
        return CompileSummary(False, False, engine, log_path, note), note
    cmd = [latexmk, "-pdf", "-g", "-interaction=nonstopmode", main_tex.name]
    proc = subprocess.run(cmd, cwd=str(main_tex.parent), capture_output=True, text=True)
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(log, encoding="utf-8")
    note = None if proc.returncode == 0 else "latexmk reported errors"
    return CompileSummary(True, proc.returncode == 0, engine, log_path, note), log


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LLM-inspired LaTeX QA preflight")
    ap.add_argument("--plan", type=Path, required=True, help="Plan JSON used for assembly")
    ap.add_argument("--snippets", type=Path, required=True, help="Directory of generated snippets")
    ap.add_argument("--build-dir", type=Path, default=None, help="Optional build directory for reports")
    ap.add_argument("--report-dir", type=Path, default=None, help="Override QA report directory")
    ap.add_argument("--attempt-compile", action="store_true", help="Run latexmk as part of QA")
    ap.add_argument("--max-passes", type=int, default=1, help="Maximum auto-fix passes per environment")
    ap.add_argument("--main-tex", type=Path, default=None, help="Optional path to an existing main.tex to compile")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    report = run_preflight(
        args.plan,
        args.snippets,
        build_dir=args.build_dir,
        report_dir=args.report_dir,
        attempt_compile=args.attempt_compile,
        max_passes=args.max_passes,
        main_tex=args.main_tex,
    )
    print(json.dumps({"report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
