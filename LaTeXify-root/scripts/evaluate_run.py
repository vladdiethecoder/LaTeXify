# scripts/evaluate_run.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, Any, List, Optional
import difflib

import fitz  # PyMuPDF

from scripts.metrics_text import TextScores
from scripts.consensus import save_consensus_for_run

def _load_latest_run_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    runs = sorted((Path("dev") / "runs").glob("*"))
    if not runs:
        raise SystemExit("No runs in dev/runs/.")
    return runs[-1]

def _extract_pdf_plaintext(pdf_path: Path) -> List[str]:
    """Return list[str] page_texts extracted from the source PDF."""
    doc = fitz.open(str(pdf_path))
    out = []
    for i in range(len(doc)):
        # plain text; we don't try to preserve layout here for the stand-in
        out.append(doc[i].get_text("text") or "")
    return out

def _read_summary(run_dir: Path) -> Dict[str, Any]:
    s = run_dir / "summary.json"
    if not s.exists():
        raise SystemExit(f"Missing summary.json in {run_dir}")
    return json.loads(s.read_text(encoding="utf-8"))

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def _best_guess_ref_for_page(pdf_texts: List[str], page_png_name: str) -> str:
    """
    Our ocr pipeline names pages as page-0001.png … page-XXXX.png.
    Map to 1-based page index.
    """
    try:
        num = int(page_png_name.replace("page-", "").replace(".png", ""))
        idx = max(1, num) - 1
        return pdf_texts[idx] if 0 <= idx < len(pdf_texts) else ""
    except Exception:
        return ""

def _write_diff(run_dir: Path, page: str, ref: str, hyp: str, model: str) -> Path:
    out_dir = run_dir / "diffs" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / page.replace(".png", ".diff.txt")
    d = difflib.unified_diff(
        ref.splitlines(keepends=True),
        hyp.splitlines(keepends=True),
        fromfile="reference",
        tofile=model,
        lineterm=""
    )
    p.write_text("".join(d), encoding="utf-8")
    return p

def _summarize_issue(page: str, model: str, scores: TextScores, ref: str, hyp: str) -> Dict[str, Any]:
    issues = []
    if scores.wer >= 0.4:
        issues.append("High WER")
    if scores.cer >= 0.25:
        issues.append("High CER")
    if scores.jaccard_unigram <= 0.5:
        issues.append("Low unigram overlap")
    if scores.bleuish <= 0.2:
        issues.append("Very low BLEU-ish")
    if len(hyp.strip()) < max(20, int(0.2 * scores.ref_len_chars)):
        issues.append("Hypothesis looks too short (truncation?)")
    if len(hyp.strip()) > int(2.5 * scores.ref_len_chars) and scores.bleuish < 0.3:
        issues.append("Hypothesis looks too long (hallucination?)")
    return {
        "page": page,
        "model": model,
        "summary": "; ".join(issues) if issues else "OK-ish",
        "scores": scores.__dict__,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None, help="Run folder (defaults to latest in dev/runs/)")
    ap.add_argument("--pdf", type=str, default=None, help="If summary.json lacks pdf path, provide it here")
    ap.add_argument("--write_csv", action="store_true", help="Emit evaluation.csv")
    args = ap.parse_args()

    run_dir = _load_latest_run_dir(args.run_dir)
    summary = _read_summary(run_dir)

    # Source PDF
    pdf_path = Path(summary.get("pdf") or args.pdf or "")
    if not pdf_path.exists():
        raise SystemExit("No source PDF. Pass --pdf <file> or ensure summary.json recorded it.")
    pdf_texts = _extract_pdf_plaintext(pdf_path)

    # Build consensus files (stand-in heuristic) and keep paths handy
    consensus_paths = save_consensus_for_run(run_dir, summary.get("per_page", []))
    page_to_cons_path = {p.name.replace(".md", ".png"): p for p in consensus_paths}

    # Prepare outputs
    rows: List[Dict[str, Any]] = []
    issues_rollup: List[Dict[str, Any]] = []

    per_page: List[Dict[str, Any]] = summary.get("per_page", [])
    for page_rec in per_page:
        page = page_rec.get("page", "")
        ref_text = _best_guess_ref_for_page(pdf_texts, page)
        # Evaluate each model
        for r in page_rec.get("results", []):
            model = r.get("model", "unknown")
            md = r.get("out_md")
            if not md:
                # record as missing output
                rows.append({
                    "page": page, "model": model, "status": "no-output",
                    "wer": "", "cer": "", "lev": "", "jaccard_uni": "", "bleuish": "",
                    "ref_len_chars": len(ref_text), "hyp_len_chars": 0,
                    "ref_len_words": len(ref_text.split()), "hyp_len_words": 0,
                    "diff_path": ""
                })
                continue
            hyp_text = _read_text(Path(md))
            scores = TextScores.compute(hyp_text, ref_text)
            diff_p = _write_diff(run_dir, page, ref_text, hyp_text, model)
            rows.append({
                "page": page, "model": model, "status": "ok",
                "wer": scores.wer, "cer": scores.cer, "lev": scores.lev_dist,
                "jaccard_uni": scores.jaccard_unigram, "bleuish": scores.bleuish,
                "ref_len_chars": scores.ref_len_chars, "hyp_len_chars": scores.hyp_len_chars,
                "ref_len_words": scores.ref_len_words, "hyp_len_words": scores.hyp_len_words,
                "diff_path": str(diff_p)
            })
            issues_rollup.append(_summarize_issue(page, model, scores, ref_text, hyp_text))

        # Evaluate consensus (if present)
        cons_path = page_to_cons_path.get(page)
        if cons_path and cons_path.exists():
            cons_text = _read_text(cons_path)
            model = "_consensus_"
            scores = TextScores.compute(cons_text, ref_text)
            diff_p = _write_diff(run_dir, page, ref_text, cons_text, model)
            rows.append({
                "page": page, "model": model, "status": "ok",
                "wer": scores.wer, "cer": scores.cer, "lev": scores.lev_dist,
                "jaccard_uni": scores.jaccard_unigram, "bleuish": scores.bleuish,
                "ref_len_chars": scores.ref_len_chars, "hyp_len_chars": scores.hyp_len_chars,
                "ref_len_words": scores.ref_len_words, "hyp_len_words": scores.hyp_len_words,
                "diff_path": str(diff_p)
            })
            issues_rollup.append(_summarize_issue(page, model, scores, ref_text, cons_text))

    # Write CSV
    if args.write_csv:
        csv_path = run_dir / "evaluation.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "page","model","status",
                "wer","cer","lev","jaccard_uni","bleuish",
                "ref_len_chars","hyp_len_chars","ref_len_words","hyp_len_words",
                "diff_path"
            ])
            w.writeheader()
            for row in rows:
                w.writerow(row)

    # Human roll-up
    report = run_dir / "evaluation_report.md"
    total = len([r for r in rows if r["status"] == "ok"])
    bad_wer = len([r for r in rows if r["status"] == "ok" and isinstance(r["wer"], (int, float)) and r["wer"] >= 0.4])
    bad_cer = len([r for r in rows if r["status"] == "ok" and isinstance(r["cer"], (int, float)) and r["cer"] >= 0.25])
    report.write_text(
        "# OCR Evaluation (stand-in)\n\n"
        f"- Run dir: `{run_dir}`\n"
        f"- Source PDF: `{pdf_path}`\n"
        f"- Systems evaluated (incl. consensus): {sorted({r['model'] for r in rows})}\n"
        f"- Evaluated rows: {total}\n"
        f"- High WER (>= 0.4): {bad_wer}\n"
        f"- High CER (>= 0.25): {bad_cer}\n\n"
        "See `evaluation.csv` for details and `diffs/` for unified diffs per page.\n",
        encoding="utf-8"
    )

    # Machine-friendly issues for future agent
    issues_path = run_dir / "issues_summary.json"
    issues_path.write_text(json.dumps(issues_rollup, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {run_dir/'evaluation.csv'}")
    print(f"Wrote {report}")
    print(f"Wrote {issues_path}")


if __name__ == "__main__":
    main()
