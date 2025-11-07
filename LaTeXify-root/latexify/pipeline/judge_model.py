from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .model_backends import LlamaCppBackend, LlamaCppConfig
except Exception:  # pragma: no cover - llama backend optional
    LlamaCppBackend = None  # type: ignore
    LlamaCppConfig = None  # type: ignore

_WS = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    return _WS.sub(" ", (value or "")).strip()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            delete = cur[j - 1] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _cer(a: str, b: str) -> float:
    norm_a = _normalize_text(a)
    norm_b = _normalize_text(b)
    if not norm_a and not norm_b:
        return 0.0
    dist = _levenshtein(norm_a, norm_b)
    denom = max(1, len(norm_a), len(norm_b))
    return dist / denom


@dataclass
class CouncilRecord:
    backend: str
    chunk_id: str
    page_index: int
    text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilChunk:
    chunk_id: str
    page_index: int
    page_name: str
    block_type: str
    records: List[CouncilRecord] = field(default_factory=list)
    consensus: Dict[str, Any] | None = None


@dataclass
class JudgeVerdict:
    chunk_id: str
    page_index: int
    page_name: str
    block_type: str
    best_backend: str
    golden_text: str
    best_confidence: float
    artifact_notes: List[str]
    relevance: float
    judge_confidence: float
    flagged: bool
    flag_reasons: List[str]
    ocr_outputs: Dict[str, str]
    latex_consensus: Optional[str]
    content_type: Optional[str] = None

    def to_block_record(self) -> Dict[str, Any]:
        return {
            "block_id": self.chunk_id,
            "page_index": self.page_index,
            "page": self.page_name,
            "block_type": self.block_type,
            "text": self.golden_text,
            "text_backend": self.best_backend,
            "ocr_outputs": self.ocr_outputs,
            "flagged": self.flagged,
            "flag_reasons": self.flag_reasons,
            "agreement_score": self.relevance,
            "latex_consensus": self.latex_consensus,
            "content_type": self.content_type,
            "judge": {
                "golden_snippet": {
                    "backend": self.best_backend,
                    "confidence": self.best_confidence,
                    "text": self.golden_text,
                },
                "artifact_notes": self.artifact_notes,
                "relevance": self.relevance,
                "confidence": self.judge_confidence,
            },
        }


class JudgeLLM:
    def __init__(
        self,
        model_path: Optional[Path],
        *,
        n_ctx: int = 8192,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend: Optional[LlamaCppBackend] = None
        if model_path is not None and LlamaCppBackend and LlamaCppConfig:
            cfg = LlamaCppConfig(model_path=model_path, n_ctx=n_ctx, n_batch=512, seed=1337)
            self.backend = LlamaCppBackend(cfg)

    def judge(self, chunk: CouncilChunk) -> Optional[JudgeVerdict]:
        if self.backend is None:
            return None
        prompt = self._build_prompt(chunk)
        response = self.backend.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        try:
            data = json.loads(response.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return _verdict_from_model_dict(chunk, data)

    def _build_prompt(self, chunk: CouncilChunk) -> str:
        payload = {
            "task": "judge-council-chunk",
            "chunk_id": chunk.chunk_id,
            "page_index": chunk.page_index,
            "block_type": chunk.block_type,
            "records": [
                {
                    "backend": rec.backend,
                    "confidence": rec.confidence,
                    "text": rec.text,
                    "metadata": rec.metadata,
                }
                for rec in chunk.records
            ],
        }
        inline_context = chunk.consensus or {}
        content_hint = inline_context.get("content_type")
        hint_line = (
            f"Content hint: {content_hint}." if content_hint else "Content hint: none provided."
        )
        instructions = (
            "You are a scholarly LaTeX judge. Study the council snippets and produce the single best "
            "golden snippet. Ensure math/chemistry/code are correct and cite any uncertainties. "
            "Return JSON with keys: golden_snippet{text,backend}, relevance (0-1), artifacts[], flagged, notes[].\n"
            + hint_line
        )
        return json.dumps({"instructions": instructions, "payload": payload}, ensure_ascii=False)


def _verdict_from_model_dict(chunk: CouncilChunk, data: Dict[str, Any]) -> Optional[JudgeVerdict]:
    golden = data.get("golden_snippet") or {}
    text = _normalize_text(golden.get("text", ""))
    backend = golden.get("backend") or chunk.records[0].backend
    artifacts = data.get("artifacts") or []
    if not isinstance(artifacts, list):
        artifacts = [str(artifacts)]
    relevance = float(data.get("relevance") or 0)
    flagged = bool(data.get("flagged")) or (relevance < 0.4)
    notes = data.get("notes") or []
    if not isinstance(notes, list):
        notes = [str(notes)]
    ocr_outputs = {rec.backend: rec.text for rec in chunk.records}
    latex_consensus = _extract_latex(chunk.records)
    return JudgeVerdict(
        chunk_id=chunk.chunk_id,
        page_index=chunk.page_index,
        page_name=chunk.page_name,
        block_type=chunk.block_type,
        best_backend=backend,
        golden_text=text,
        best_confidence=1.0,
        artifact_notes=[str(a) for a in artifacts],
        relevance=max(0.0, min(1.0, relevance)),
        judge_confidence=0.7,
        flagged=flagged,
        flag_reasons=[str(n) for n in notes],
        ocr_outputs=ocr_outputs,
        latex_consensus=latex_consensus,
        content_type=(chunk.consensus or {}).get("content_type"),
    )


class JudgeModel:
    def __init__(self, llm: Optional[JudgeLLM] = None):
        self.llm = llm

    def evaluate_chunk(self, chunk: CouncilChunk) -> JudgeVerdict:
        verdict = None
        if self.llm:
            verdict = self.llm.judge(chunk)
        if verdict is None:
            verdict = self._heuristic(chunk)
        verdict.golden_text = self._format_snippet(chunk, verdict.golden_text)
        verdict.content_type = (chunk.consensus or {}).get("content_type") or chunk.block_type
        return verdict

    def _heuristic(self, chunk: CouncilChunk) -> JudgeVerdict:
        records = chunk.records or [
            CouncilRecord(
                backend="unknown",
                chunk_id=chunk.chunk_id,
                page_index=chunk.page_index,
                text="",
                confidence=0.0,
            )
        ]
        best = max(records, key=lambda rec: (rec.confidence, len(_normalize_text(rec.text))))
        others = [rec for rec in records if rec is not best]
        artifact_notes: List[str] = []
        if not _normalize_text(best.text):
            artifact_notes.append("Empty snippet")
        if any("warning" in rec.metadata for rec in records):
            artifact_notes.append("Backend warnings present")
        lengths = [len(_normalize_text(rec.text)) for rec in records]
        if lengths and max(lengths) > 80 and len(_normalize_text(best.text)) < 0.25 * max(lengths):
            artifact_notes.append("Selected snippet much shorter than alternatives")
        pair_scores = [1.0 - _cer(best.text, rec.text) for rec in others if rec.text]
        relevance = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
        flagged = relevance < 0.6 or any("Empty" in note for note in artifact_notes)
        flag_reasons = list(artifact_notes)
        ocr_outputs = {rec.backend: rec.text for rec in records}
        latex_consensus = _extract_latex(records)
        return JudgeVerdict(
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            page_name=chunk.page_name,
            block_type=chunk.block_type,
            best_backend=best.backend,
            golden_text=best.text.strip(),
            best_confidence=float(best.confidence),
            artifact_notes=artifact_notes,
            relevance=max(0.0, min(1.0, relevance)),
            judge_confidence=0.55,
            flagged=flagged,
            flag_reasons=flag_reasons,
            ocr_outputs=ocr_outputs,
            latex_consensus=latex_consensus,
            content_type=(chunk.consensus or {}).get("content_type"),
        )

    def _format_snippet(self, chunk: CouncilChunk, text: str) -> str:
        ctype = (chunk.consensus or {}).get("content_type") or (chunk.block_type or "text")
        formatter = ctype.lower()
        if formatter in {"table", "table_with_text"}:
            return build_table_snippet(text)
        if formatter in {"figure_with_text", "figure", "graph"}:
            return build_figure_snippet(text, chunk.chunk_id)
        if formatter == "math":
            body = text.strip()
            if not body.startswith("$") and not body.startswith("\\"):
                body = f"${body}$"
            return body + "\n"
        if formatter == "code":
            return "\\begin{verbatim}\n" + text.strip() + "\n\\end{verbatim}\n"
        if formatter == "chem":
            return f"% chem snippet\n{ text.strip() }\n"
        return text.strip() + ("\n" if not text.endswith("\n") else "")


def _extract_latex(records: Iterable[CouncilRecord]) -> Optional[str]:
    for rec in records:
        if rec.backend.lower().startswith("nougat") and "\\" in rec.text:
            return rec.text.strip()
    for rec in records:
        if "$" in rec.text or "\\[" in rec.text:
            return rec.text.strip()
    return None


def build_table_snippet(text: str) -> str:
    rows = _parse_table_rows(text)
    if not rows:
        return text.strip() + "\n"
    width = max(len(row) for row in rows)
    header = rows[0] if rows else [""] * width
    body = rows[1:] or [[""] * width]
    cols = "l" * width
    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Auto-generated table}",
        f"  \\begin{{tabular}}{{{cols}}}",
        "    \\toprule",
        "    " + " & ".join(header) + r" \\",
        "    \\midrule",
    ]
    for row in body:
        padded = row + [""] * (width - len(row))
        lines.append("    " + " & ".join(padded) + r" \\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_figure_snippet(caption_text: str, chunk_id: str) -> str:
    caption = caption_text.strip().split("\n")[0]
    label = _slugify(chunk_id)
    return "\n".join(
        [
            "\\begin{figure}[ht]",
            "  \\centering",
            "  % TODO: replace placeholder with reconstructed graphic",
            "  \\includegraphics[width=0.85\\linewidth]{assets/figure-placeholder.pdf}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{fig:{label}}}",
            "\\end{figure}",
            "",
        ]
    )


def _parse_table_rows(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in text.splitlines():
        cleaned = line.strip().strip("|")
        if not cleaned:
            continue
        if "|" in line:
            cells = [c.strip() for c in cleaned.split("|")]
        else:
            cells = [c.strip() for c in re.split(r"\s{2,}", cleaned) if c.strip()]
        if cells:
            rows.append(cells)
    return rows


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "chunk"


def load_council_chunks(run_dir: Path, consensus_dir: Optional[Path] = None) -> List[CouncilChunk]:
    outputs_root = run_dir / "outputs"
    if not outputs_root.exists():
        raise FileNotFoundError(f"No outputs directory in {run_dir}")
    chunk_index: Dict[str, CouncilChunk] = {}
    consensus_lookup = _load_consensus(consensus_dir) if consensus_dir else {}
    for backend_dir in outputs_root.iterdir():
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name
        for json_file in backend_dir.glob("*.json"):
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            chunk_id = payload.get("chunk_id") or json_file.stem
            page_index = int(payload.get("page_index") or 0)
            text = payload.get("text") or ""
            confidence = float(payload.get("confidence") or 0.5)
            metadata = payload.get("metadata") or {}
            page_name = metadata.get("page_name") or f"page-{page_index + 1:04d}.png"
            block_type = metadata.get("block_type") or "text"
            chunk = chunk_index.setdefault(
                chunk_id,
                CouncilChunk(
                    chunk_id=chunk_id,
                    page_index=page_index,
                    page_name=page_name,
                    block_type=block_type,
                    consensus=consensus_lookup.get(chunk_id),
                ),
            )
            chunk.records.append(
                CouncilRecord(
                    backend=backend,
                    chunk_id=chunk_id,
                    page_index=page_index,
                    text=text,
                    confidence=confidence,
                    metadata=metadata,
                )
            )
    for chunk_id, chunk in chunk_index.items():
        if chunk.consensus and chunk.consensus.get("content_type"):
            chunk.block_type = chunk.consensus.get("content_type") or chunk.block_type
    return list(chunk_index.values())


def _load_consensus(consensus_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not consensus_dir or not consensus_dir.exists():
        return lookup
    for path in consensus_dir.glob("*.json"):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        chunk_id = data.get("chunk_id") or path.stem
        lookup[str(chunk_id)] = data
    return lookup


def run_judge(
    run_dir: Path,
    *,
    out_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    consensus_dir: Optional[Path] = None,
) -> Path:
    consensus_dir = consensus_dir or (run_dir / "consensus")
    chunks = load_council_chunks(run_dir, consensus_dir if consensus_dir and consensus_dir.exists() else None)
    llm = JudgeLLM(model_path) if model_path else None
    judge = JudgeModel(llm)
    verdicts = [judge.evaluate_chunk(chunk).to_block_record() for chunk in chunks]
    out = out_path or (run_dir / "blocks_refined.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in verdicts:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    golden_dir = run_dir / "golden_snippets"
    golden_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"snippets": []}
    for record in verdicts:
        snippet_path = golden_dir / f"{record['block_id']}.tex"
        snippet_path.write_text(record["text"], encoding="utf-8")
        manifest["snippets"].append({
            "block_id": record["block_id"],
            "path": str(snippet_path.relative_to(run_dir)),
            "content_type": record.get("content_type"),
            "flagged": record.get("flagged"),
        })
    (golden_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Judge model: consolidate council outputs into golden snippets")
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory containing outputs/<backend>")
    ap.add_argument("--out", type=Path, default=None, help="Destination blocks_refined.jsonl path")
    ap.add_argument("--model-path", type=Path, default=None, help="Optional GGUF model path for llama.cpp judge")
    ap.add_argument("--consensus-dir", type=Path, default=None, help="Optional consensus directory (defaults to run_dir/consensus)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    out = run_judge(
        args.run_dir,
        out_path=args.out,
        model_path=args.model_path,
        consensus_dir=args.consensus_dir,
    )
    print(json.dumps({"run_dir": str(args.run_dir), "blocks_refined": str(out)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
