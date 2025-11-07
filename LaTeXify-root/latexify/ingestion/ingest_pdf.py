from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence

from pypdf import PdfReader


DEFAULT_ASSETS_DIR = Path("build/assets")
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert academic document reader. Describe what the provided chunk contains "
    "in polished Markdown. Focus on clarity and accuracy."
)
DEFAULT_FLORENCE_PROMPT = (
    "You are Florence-2 running in OCR-with-region mode. Return JSON with an array `items`, "
    "each containing `text` and normalized bounding box coordinates [x0,y0,x1,y1]."
)

PLACEHOLDER_SNIPPETS = {
    "",
    "(no text extracted)",
    "\\[ \\text{No equations detected} \\]",
}


def _encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [p.strip() for p in parts if p.strip()]


CHEM_RX = re.compile(r"\b(?:[A-Z][a-z]?\d{0,3}){2,}\b")
MATH_RX = re.compile(r"(\$\$|\\\[|\\\(|\\begin\{equation|\\sum|\\int)")
TABLE_RX = re.compile(r"\\begin\{tabular|\|.*\|")
CODE_RX = re.compile(r"```|\bclass\b|\bdef \b|\bfor \(|\{\s*$", re.MULTILINE)
FIG_RX = re.compile(r"figure|diagram|plot|graph|image", re.IGNORECASE)


def classify_snippet(text: str) -> Dict[str, Any]:
    snippet = (text or "").strip()
    lower = snippet.lower()
    if not snippet:
        return {"content_type": "text", "confidence": 0.2, "explanation": "Empty snippet; defaulting to text."}
    if TABLE_RX.search(snippet):
        return {"content_type": "table", "confidence": 0.9, "explanation": "Detected tabular delimiters."}
    if MATH_RX.search(snippet):
        return {"content_type": "math", "confidence": 0.9, "explanation": "LaTeX math markers present."}
    if CODE_RX.search(snippet):
        return {"content_type": "code", "confidence": 0.75, "explanation": "Code fences/keywords detected."}
    if CHEM_RX.search(snippet):
        return {"content_type": "chem", "confidence": 0.7, "explanation": "Chemical formula pattern detected."}
    if FIG_RX.search(lower):
        return {"content_type": "figure_with_text", "confidence": 0.65, "explanation": "Figure keywords detected."}
    return {"content_type": "text", "confidence": 0.5, "explanation": "Defaulted to prose."}


def _is_meaningful_text(text: str) -> bool:
    stripped = (text or "").strip()
    if stripped in PLACEHOLDER_SNIPPETS:
        return False
    if len(stripped) < 4:
        return False
    letters = sum(ch.isalpha() for ch in stripped)
    return letters >= 2


def _normalize_endpoint(base: str) -> str:
    base = (base or "").strip()
    if not base:
        return ""
    if base.rstrip("/").endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base.rstrip("/") + "/chat/completions"
    return base.rstrip("/") + "/v1/chat/completions"


@dataclass
class LayoutChunk:
    chunk_id: str
    page_index: int
    text: str
    page_name: str
    bbox: List[float] | None = None
    image_path: Path | None = None


@dataclass
class CouncilOutput:
    backend: str
    chunk_id: str
    page_index: int
    text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCouncilBackend:
    name: str = "base"

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        raise NotImplementedError


class VisionChatBackend(BaseCouncilBackend):
    def __init__(
        self,
        name: str,
        endpoint: str | None,
        model: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        self.name = name
        self.endpoint = _normalize_endpoint(endpoint or "")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        if self.endpoint:
            try:
                text, meta = await asyncio.get_running_loop().run_in_executor(
                    None, self._call_endpoint, chunk
                )
                return CouncilOutput(
                    backend=self.name,
                    chunk_id=chunk.chunk_id,
                    page_index=chunk.page_index,
                    text=text.strip(),
                    confidence=meta.get("confidence", 0.78),
                    metadata=meta,
                )
            except Exception as exc:
                meta = {"warning": f"{self.name} request failed: {exc}"}
        else:
            meta = {"warning": f"{self.name} endpoint not configured; using fallback"}
        fallback = self._fallback_text(chunk)
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=fallback,
            confidence=0.35,
            metadata=meta,
        )

    def _build_messages(self, chunk: LayoutChunk) -> List[Dict[str, Any]]:
        user_parts: List[Dict[str, Any]] = []

        def add_text(txt: str) -> None:
            user_parts.append({"type": "text", "text": txt})

        add_text(
            "\n".join(
                [
                    f"Page {chunk.page_index + 1} chunk `{chunk.chunk_id}`.",
                    "Describe the text faithfully and keep math inline when possible.",
                ]
            )
        )
        if chunk.image_path and chunk.image_path.exists():
            add_text("[Attached chunk image]")
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image(chunk.image_path), "detail": "high"},
                }
            )
        if chunk.text:
            add_text("Raw text:")
            add_text(chunk.text[:4000])
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_parts}]

    def _call_endpoint(self, chunk: LayoutChunk) -> tuple[str, Dict[str, Any]]:
        messages = self._build_messages(chunk)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        headers: MutableMapping[str, str] = {"Content-Type": "application/json"}
        api_key = os.environ.get("LATEXIFY_COUNCIL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.name} returned no choices")
        message = choices[0].get("message") or {}
        text = message.get("content", "")
        meta = {
            "finish_reason": choices[0].get("finish_reason"),
            "usage": parsed.get("usage"),
            "raw_response": text,
        }
        cleaned = self._extract_text(text)
        return cleaned, meta

    def _extract_text(self, payload: str) -> str:
        return payload.strip()

    def _fallback_text(self, chunk: LayoutChunk) -> str:
        lines = textwrap.wrap(chunk.text or "", width=80)
        snippet = "\n".join(lines[:6]).strip()
        return snippet or "(no text extracted)"


class FlorenceBackend(VisionChatBackend):
    def __init__(self, endpoint: str | None, model: str = "Florence-2") -> None:
        super().__init__(
            name="florence2",
            endpoint=endpoint,
            model=model,
            system_prompt=DEFAULT_FLORENCE_PROMPT,
            temperature=0.0,
            max_tokens=700,
        )

    def _extract_text(self, payload: str) -> str:
        try:
            data = json.loads(payload)
            items = data.get("items") or []
            lines = []
            for item in items:
                bbox = item.get("bbox") or [0, 0, 1, 1]
                lines.append(f"[{bbox}] {item.get('text','').strip()}")
            return "\n".join(lines)
        except Exception:
            return payload.strip()

    def _fallback_text(self, chunk: LayoutChunk) -> str:
        norm_bbox = chunk.bbox or [0.0, 0.0, 1.0, 1.0]
        return json.dumps(
            {"items": [{"text": chunk.text.strip(), "bbox": norm_bbox, "source": "heuristic"}]},
            ensure_ascii=False,
        )


class MinerUBackend(BaseCouncilBackend):
    name = "mineru"

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        markdown = self._to_table(chunk.text)
        metadata = {"columns": markdown["columns"], "rows": len(markdown["rows"])}
        table_md = self._render_table(markdown)
        row_count = len(markdown["rows"])
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=table_md,
            confidence=0.6 if row_count > 1 else 0.3,
            metadata=metadata,
        )

    def _to_table(self, text: str) -> Dict[str, Any]:
        rows: List[List[str]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            cells = re.split(r"\s{3,}|\t|\s\|\s", line.strip())
            if len(cells) >= 2:
                rows.append(cells)
        if not rows:
            return {"columns": [], "rows": []}
        width = max(len(r) for r in rows)
        normalized = [r + [""] * (width - len(r)) for r in rows]
        header = normalized[0]
        body = normalized[1:] or [[""] * width]
        return {"columns": header, "rows": body}

    def _render_table(self, table: Dict[str, Any]) -> str:
        columns: List[str] = table["columns"] or ["Column 1", "Column 2"]
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [header, divider]
        for row in table["rows"]:
            lines.append("| " + " | ".join(row[: len(columns)]) + " |")
        if len(lines) == 2:
            lines.append("| " + " | ".join("" for _ in columns) + " |")
        return "\n".join(lines)


class NougatBackend(BaseCouncilBackend):
    name = "nougat"
    EQUATION_RX = re.compile(r"([A-Za-z0-9\s]+=[^\\n]+)")

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        latex = self._extract_equations(chunk.text)
        meta = {"equation_count": latex.count("\\[")}
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=latex or "\\[ \\text{No equations detected} \\]",
            confidence=0.55 if latex else 0.2,
            metadata=meta,
        )

    def _extract_equations(self, text: str) -> str:
        matches = self.EQUATION_RX.findall(text)
        if not matches:
            return ""
        lines = [f"\\[{m.strip()}\\]" for m in matches[:6]]
        return "\n".join(lines)


class InternVLBackend(VisionChatBackend):
    def __init__(self, endpoint: str | None, model: str = "InternVL-3.5-14B") -> None:
        super().__init__(
            name="internvl",
            endpoint=endpoint,
            model=model,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=700,
        )


class GenericOCRBackend(BaseCouncilBackend):
    """Lightweight permissive fallback that reuses raw PDF text."""

    name = "generic_ocr"
    permissive = True

    async def process(self, chunk: LayoutChunk) -> CouncilOutput:
        text = (chunk.text or "").strip()
        if not text:
            text = f"[permissive-fallback chunk={chunk.chunk_id}]"
        metadata = {
            "fallback_used": True,
            "reason": "permissive-backup",
            "source": self.name,
        }
        return CouncilOutput(
            backend=self.name,
            chunk_id=chunk.chunk_id,
            page_index=chunk.page_index,
            text=text,
            confidence=0.42,
            metadata=metadata,
        )


class CouncilOrchestrator:
    def __init__(self, backends: Sequence[BaseCouncilBackend], run_dir: Path) -> None:
        self.backends = list(backends)
        self.run_dir = run_dir
        self.records: List[Dict[str, Any]] = []
        self.records_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
        self.permissive_backends = [backend.name for backend in self.backends if getattr(backend, "permissive", False)]
        self.stats: Dict[str, int] = {
            "backend_warnings": 0,
            "fallback_hits": 0,
            "permissive_hits": 0,
            "consensus_recovered": 0,
            "empty_consensus": 0,
        }
        self.recoveries: List[Dict[str, Any]] = []

    async def process_chunks(self, chunks: Sequence[LayoutChunk]) -> None:
        outputs_root = self.run_dir / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            tasks = [backend.process(chunk) for backend in self.backends]
            results = await asyncio.gather(*tasks)
            for result in results:
                self._persist(result)

    def _persist(self, result: CouncilOutput) -> None:
        backend_dir = self.run_dir / "outputs" / result.backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        metadata = result.metadata or {}
        payload = {
            "backend": result.backend,
            "chunk_id": result.chunk_id,
            "page_index": result.page_index,
            "text": result.text,
            "confidence": result.confidence,
            "metadata": metadata,
        }
        out_path = backend_dir / f"{result.chunk_id}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.records.append(
            {
                "backend": result.backend,
                "chunk_id": result.chunk_id,
                "page_index": result.page_index,
                "path": str(out_path),
                "confidence": result.confidence,
                "metadata": metadata,
            }
        )
        self.records_by_chunk.setdefault(result.chunk_id, []).append(payload)
        if metadata.get("warning"):
            self.stats["backend_warnings"] += 1
        if metadata.get("fallback_used"):
            self.stats["fallback_hits"] += 1
            self.recoveries.append(
                {
                    "chunk_id": result.chunk_id,
                    "backend": result.backend,
                    "reason": metadata.get("reason") or metadata.get("warning") or "fallback",
                }
            )
        if metadata.get("source") == "generic_ocr":
            self.stats["permissive_hits"] += 1

    def write_manifest(self) -> None:
        council_dir = self.run_dir / "council"
        council_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": _now_ts(),
            "records": self.records,
            "backends": [backend.name for backend in self.backends],
        }
        (council_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def write_consensus(self) -> Path:
        consensus_dir = self.run_dir / "consensus"
        consensus_dir.mkdir(parents=True, exist_ok=True)
        chunk_ids: List[str] = []
        for chunk_id, outputs in sorted(self.records_by_chunk.items()):
            chunk_ids.append(chunk_id)
            merged = self._merge_chunk(chunk_id, outputs)
            (consensus_dir / f"{chunk_id}.json").write_text(
                json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (consensus_dir / f"{chunk_id}.txt").write_text(merged["text"], encoding="utf-8")
        (consensus_dir / "manifest.json").write_text(
            json.dumps({"chunks": chunk_ids, "count": len(chunk_ids)}, indent=2),
            encoding="utf-8",
        )
        return consensus_dir

    def write_resilience_report(self) -> Path:
        report = {
            "created_at": _now_ts(),
            "chunks": len(self.records_by_chunk),
            "permissive_backends": self.permissive_backends,
            "stats": {k: int(v) for k, v in self.stats.items()},
            "recoveries": self.recoveries,
        }
        path = self.run_dir / "resilience_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path

    def _merge_chunk(self, chunk_id: str, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = self._select_best(outputs)
        selected = best
        if not _is_meaningful_text(best["text"]):
            fallback = next((entry for entry in outputs if _is_meaningful_text(entry["text"])), None)
            if fallback:
                selected = fallback
                self.stats["consensus_recovered"] += 1
                self.recoveries.append(
                    {
                        "chunk_id": chunk_id,
                        "backend": fallback["backend"],
                        "reason": "consensus_replacement",
                    }
                )
            else:
                self.stats["empty_consensus"] += 1
        ocr_outputs = {entry["backend"]: entry["text"] for entry in outputs}
        classification = classify_snippet(selected["text"])
        return {
            "chunk_id": chunk_id,
            "text": selected["text"],
            "content_type": classification["content_type"],
            "classification_confidence": classification["confidence"],
            "classification_explanation": classification["explanation"],
            "source_backend": selected["backend"],
            "supporting_backends": [entry["backend"] for entry in outputs if entry is not selected],
            "ocr_outputs": ocr_outputs,
        }

    def _select_best(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = {"internvl": 0.9, "florence2": 0.6, "mineru": 0.85, "nougat": 0.8}

        def score(entry: Dict[str, Any]) -> float:
            backend = entry["backend"]
            text = entry["text"]
            base = float(entry.get("confidence") or 0.5) + weights.get(backend, 0.2)
            if backend == "mineru" and ("|" in text or "\n" in text):
                base += 0.2
            if backend == "nougat" and ("\\[" in text or "$$" in text):
                base += 0.2
            if not text.strip():
                base -= 0.5
            return base

        return max(outputs, key=score)


def _extract_pdf_text(pdf: Path, max_pages: int | None = None) -> List[str]:
    reader = PdfReader(str(pdf))
    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        if max_pages is not None and idx >= max_pages:
            break
        pages.append(page.extract_text() or "")
    return pages


def _chunk_pages(
    pages: Sequence[str],
    *,
    chunk_chars: int = 1100,
    min_chars: int = 320,
) -> List[LayoutChunk]:
    chunks: List[LayoutChunk] = []
    for page_index, text in enumerate(pages):
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text]
        buffer = []
        counter = 0
        for para in paragraphs:
            buffer.append(para)
            joined = "\n\n".join(buffer)
            if len(joined) >= chunk_chars:
                chunk_id = f"page{page_index+1:04d}-chunk{counter+1:03d}"
                chunks.append(
                    LayoutChunk(
                        chunk_id=chunk_id,
                        page_index=page_index,
                        text=joined.strip(),
                        page_name=f"page-{page_index+1:04d}.md",
                        bbox=None,
                        image_path=None,
                    )
                )
                buffer = []
                counter += 1
        if buffer and (len(buffer[0]) >= min_chars or not chunks):
            chunk_id = f"page{page_index+1:04d}-chunk{counter+1:03d}"
            chunks.append(
                LayoutChunk(
                    chunk_id=chunk_id,
                    page_index=page_index,
                    text="\n\n".join(buffer).strip(),
                    page_name=f"page-{page_index+1:04d}.md",
                )
            )
    return chunks


def ingest_pdf(
    pdf: Path,
    run_dir: Path,
    *,
    assets_dir: Path | None = None,
    max_pages: int | None = None,
    chunk_chars: int = 1100,
    internvl_endpoint: str | None = None,
    florence_endpoint: str | None = None,
    pages_override: Sequence[str] | None = None,
    permissive: bool = False,
) -> Dict[str, Any]:
    if pages_override:
        pages = list(pages_override)
    else:
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        pages = _extract_pdf_text(pdf, max_pages)
    run_dir.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_pages(pages, chunk_chars=chunk_chars)
    internvl = InternVLBackend(internvl_endpoint)
    florence = FlorenceBackend(florence_endpoint)
    mineru = MinerUBackend()
    nougat = NougatBackend()
    backends: List[BaseCouncilBackend] = [internvl, florence, mineru, nougat]
    if permissive:
        backends.append(GenericOCRBackend())
    orchestrator = CouncilOrchestrator(backends, run_dir)
    asyncio.run(orchestrator.process_chunks(chunks))
    orchestrator.write_manifest()
    consensus_dir = orchestrator.write_consensus()
    resilience_report = orchestrator.write_resilience_report()
    meta = {
        "source_pdf": str(pdf),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "assets_dir": str(assets_dir) if assets_dir else None,
        "backends": [backend.name for backend in backends],
        "consensus_dir": str(consensus_dir),
        "resilience_report": str(resilience_report),
        "permissive_ocr": permissive,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Multi-model ingestion council for PDFs")
    ap.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory for council outputs")
    ap.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR, help="Directory for rendered assets")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit number of pages to ingest")
    ap.add_argument("--chunk-chars", type=int, default=1100, help="Max characters per chunk before splitting")
    ap.add_argument("--internvl-endpoint", type=str, default=None, help="OpenAI-compatible endpoint for InternVL")
    ap.add_argument("--florence-endpoint", type=str, default=None, help="OpenAI-compatible endpoint for Florence-2")
    ap.add_argument("--permissive", action="store_true", help="Enable generic OCR fallback backend")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    summary = ingest_pdf(
        args.pdf,
        args.run_dir,
        assets_dir=args.assets_dir,
        max_pages=args.max_pages,
        chunk_chars=args.chunk_chars,
        internvl_endpoint=args.internvl_endpoint,
        florence_endpoint=args.florence_endpoint,
        permissive=args.permissive,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
