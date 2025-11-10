from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence

try:  # optional for async HTTP
    import aiohttp
except Exception:  # pragma: no cover - aiohttp optional
    aiohttp = None  # type: ignore[assignment]

DEFAULT_SYSTEM_PROMPT = (
    "You are a creative document layout designer. Study the provided draft "
    "content and emit a JSON layout blueprint with this shape:\n"
    "{\n"
    '  "page_layout": {"format": "...", "columns": "...", "title_area": "...", "margins": "..."},\n'
    '  "typography": {"title": "...", "headings": "...", "body": "...", "color_accents": "..."},\n'
    '  "sections": [\n'
    '    {"id": "S1", "title": "...", "goal": "...", "layout": "...", "visuals": ["figure", "table"], "notes": ["..."]}\n'
    "  ],\n"
    '  "callouts": [{"title": "...", "placement": "...", "purpose": "..."}],\n'
    '  "assets": [{"slot": "lead-figure", "type": "figure", "placement": "top", "source": "image-1"}],\n'
    '  "global_guidelines": ["..."],\n'
    '  "narrative": "One paragraph describing the overall layout flow."\n'
    "}\n"
    "Only emit valid JSON—no markdown fences or commentary."
)


FIGURE_KEYWORDS = ("figure", "diagram", "graph", "plot", "illustration", "photo", "image", "chart")
TABLE_KEYWORDS = ("table", "grid", "matrix", "schedule", "rubric")
CALL_OUT_KEYWORDS = ("key takeaway", "important", "tip", "summary", "highlight", "remember")
HEADING_RX = re.compile(r"^(chapter|section|unit|lesson|appendix|part|module)\b", re.IGNORECASE)
DOC_CLASS_RULES = (
    ("lix_textbook", 0.9, "Chapter-style headings detected."),
    ("lix_report", 0.75, "Appendix or multi-section lab-style content detected."),
    ("lix_article", 0.7, "Compact article-like layout (single/two column)."),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[... truncated {len(text) - max_chars} chars]"


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) <= 72 and (stripped.isupper() or stripped.istitle()):
        return True
    if HEADING_RX.match(stripped):
        return True
    return False


def _encode_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _normalize_endpoint(base: str) -> str:
    base = (base or "").strip()
    if not base:
        return "http://127.0.0.1:8000/v1/chat/completions"
    if base.rstrip("/").endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base.rstrip("/") + "/chat/completions"
    return base.rstrip("/") + "/v1/chat/completions"


def _extract_json_maybe(payload: str) -> Dict[str, Any] | None:
    if not payload:
        return None
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = payload[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class LayoutPlannerConfig:
    model: str = "Qwen2.5-VL-32B-Instruct"
    endpoint: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 900
    timeout: float = 60.0
    blueprint_version: str = "layout-planner/v0.1"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prefer_remote: bool = True


@dataclass
class LayoutBlueprint:
    version: str
    model_name: str
    created_at: str
    plan: Dict[str, Any]
    raw_response: str | None
    source: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model": self.model_name,
            "created_at": self.created_at,
            "plan": self.plan,
            "raw_response": self.raw_response,
            "source": self.source,
            "warnings": self.warnings,
        }

    def as_text(self) -> str:
        page_layout = self.plan.get("page_layout", {})
        sections = self.plan.get("sections", [])
        lines = [
            f"Layout blueprint ({self.model_name}) — {self.created_at}",
            "",
            f"Columns: {page_layout.get('columns', 'n/a')} | Format: {page_layout.get('format', 'n/a')}",
            f"Title area: {page_layout.get('title_area', 'n/a')}",
            "",
            "Sections:",
        ]
        for sec in sections:
            lines.append(
                f"  - {sec.get('id', '?')}: {sec.get('title', 'Untitled')} "
                f"→ {sec.get('layout', 'layout TBD')}"
            )
        if self.plan.get("global_guidelines"):
            lines.append("")
            lines.append("Guidelines:")
            for rule in self.plan["global_guidelines"]:
                lines.append(f"  • {rule}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for item in self.warnings:
                lines.append(f"  ! {item}")
        return "\n".join(lines)


class LayoutPlannerError(RuntimeError):
    pass


class LayoutPlanner:
    def __init__(self, cfg: LayoutPlannerConfig):
        self.cfg = cfg

    def generate(self, draft_text: str, images: Sequence[Path]) -> LayoutBlueprint:
        """Synchronous wrapper for generate_async with fast heuristic shortcut."""

        if not self.cfg.prefer_remote:
            fallback = self._heuristic_blueprint(draft_text or "", images)
            plan = fallback["plan"]
            self._enrich_plan(plan, draft_text or "")
            blueprint = LayoutBlueprint(
                version=self.cfg.blueprint_version,
                model_name="heuristic",
                created_at=_now_iso(),
                plan=plan,
                raw_response=None,
                source={
                    "text_chars": len(draft_text or ""),
                    "image_count": len(images),
                    "latency_ms": None,
                    "prefer_remote": False,
                },
                warnings=fallback.get("warnings", []),
            )
            return blueprint

        return asyncio.run(self.generate_async(draft_text, images))

    async def generate_async(self, draft_text: str, images: Sequence[Path]) -> LayoutBlueprint:
        draft_text = (draft_text or "").strip()
        if not draft_text and not images:
            raise LayoutPlannerError("Draft text or at least one image is required")
        fallback_task = asyncio.to_thread(self._heuristic_blueprint, draft_text, images)
        warnings: List[str] = []
        remote_json: Dict[str, Any] | None = None
        raw_response = None
        latency_ms = None
        if self.cfg.prefer_remote:
            remote_task = asyncio.create_task(self._invoke_remote_async(draft_text, images))
            fallback_result, remote_result = await asyncio.gather(fallback_task, remote_task, return_exceptions=True)
            if isinstance(fallback_result, Exception):  # pragma: no cover - defensive
                raise LayoutPlannerError(f"Heuristic planner failed: {fallback_result}") from fallback_result
            fallback = fallback_result
            if isinstance(remote_result, Exception):
                warnings.append(str(remote_result))
            else:
                remote_json, raw_response, latency_ms = remote_result
        else:
            fallback = await fallback_task
        fallback_warnings = fallback.get("warnings", [])
        if isinstance(fallback_warnings, list):
            warnings.extend(str(item) for item in fallback_warnings)
        plan = fallback["plan"]
        if remote_json:
            plan = self._merge_plan(plan, remote_json)
        if raw_response:
            plan.setdefault("model_notes", raw_response)
        plan.setdefault("stats", fallback["plan"].get("stats"))
        self._enrich_plan(plan, draft_text)
        metadata = {
            "text_chars": len(draft_text),
            "image_count": len(images),
            "latency_ms": latency_ms,
            "prefer_remote": self.cfg.prefer_remote,
        }
        blueprint = LayoutBlueprint(
            version=self.cfg.blueprint_version,
            model_name=self.cfg.model,
            created_at=_now_iso(),
            plan=plan,
            raw_response=raw_response,
            source=metadata,
            warnings=warnings,
        )
        return blueprint

    async def _invoke_remote_async(
        self,
        draft_text: str,
        images: Sequence[Path],
    ) -> tuple[Dict[str, Any] | None, str | None, float | None]:
        if aiohttp is None:
            return await asyncio.to_thread(self._invoke_remote_sync, draft_text, images)
        if not self.cfg.endpoint:
            raise LayoutPlannerError("Remote endpoint not configured")
        endpoint = _normalize_endpoint(self.cfg.endpoint)
        headers: MutableMapping[str, str] = {"Content-Type": "application/json"}
        api_key = self.cfg.api_key or os.environ.get("LATEXIFY_VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": self._build_messages(draft_text, images),
        }
        start = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.cfg.timeout)) as session:
                async with session.post(endpoint, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    body = await resp.text()
        except aiohttp.ClientError as exc:  # pragma: no cover - network failure
            raise LayoutPlannerError(f"layout planner API call failed: {exc}") from exc
        latency_ms = (time.time() - start) * 1000.0
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise LayoutPlannerError(f"layout planner API returned invalid JSON: {exc}") from exc
        choices = parsed.get("choices") or []
        if not choices:
            raise LayoutPlannerError("layout planner API returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        content_str = content if isinstance(content, str) else json.dumps(content)
        json_payload = _extract_json_maybe(content_str)
        return json_payload, content_str, latency_ms

    def _invoke_remote_sync(
        self,
        draft_text: str,
        images: Sequence[Path],
    ) -> tuple[Dict[str, Any] | None, str | None, float | None]:
        if not self.cfg.endpoint:
            raise LayoutPlannerError("Remote endpoint not configured")
        endpoint = _normalize_endpoint(self.cfg.endpoint)
        headers: MutableMapping[str, str] = {"Content-Type": "application/json"}
        api_key = self.cfg.api_key or os.environ.get("LATEXIFY_VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": self._build_messages(draft_text, images),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                body = resp.read()
        except urllib.error.URLError as exc:  # pragma: no cover - network failures not deterministic in tests
            raise LayoutPlannerError(f"layout planner API call failed: {exc}") from exc
        latency_ms = (time.time() - start) * 1000.0
        try:
            parsed = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise LayoutPlannerError(f"layout planner API returned invalid JSON: {exc}") from exc
        choices = parsed.get("choices") or []
        if not choices:
            raise LayoutPlannerError("layout planner API returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        content_str = content if isinstance(content, str) else json.dumps(content)
        json_payload = _extract_json_maybe(content_str)
        return json_payload, content_str, latency_ms

    def _build_messages(self, draft_text: str, images: Sequence[Path]) -> List[Dict[str, Any]]:
        excerpt = _truncate(draft_text, 5000) if draft_text else ""
        summary_lines = [
            "You are given draft content for a technical/academic document.",
            f"Characters supplied: {len(draft_text)}",
            f"Images supplied: {len(images)}",
        ]
        if excerpt:
            summary_lines.append("Draft excerpt:")
            summary_lines.append("<<<DRAFT>>>")
            summary_lines.append(excerpt)
            summary_lines.append("<<<END DRAFT>>>")
        summary_lines.append("Return the JSON blueprint described in the system prompt.")
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(summary_lines)}]
        for idx, image_path in enumerate(images, 1):
            if not image_path.exists():
                continue
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image(image_path), "detail": "high"},
                }
            )
            user_content.append({"type": "text", "text": f"[image {idx}: {image_path.name}]"})
        return [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _heuristic_blueprint(self, draft_text: str, images: Sequence[Path]) -> Dict[str, Any]:
        words = re.findall(r"\w+", draft_text)
        word_count = len(words)
        sections = self._split_sections(draft_text)
        columns = "two-column" if word_count > 450 or len(sections) >= 3 else "single-column"
        visuals = self._collect_visual_requests(draft_text)
        assets = self._build_asset_slots(images, visuals)
        callouts = self._build_callouts(draft_text)
        guidelines = [
            "Respect the layout planner scaffold and keep the hierarchy consistent.",
            "Use shaded sidebars for tips or highlighted learning goals when present.",
        ]
        if columns == "two-column":
            guidelines.append("Switch to two columns after the lead section for denser reading.")
        if any(v["type"] == "formula" for v in visuals):
            guidelines.append("Reserve inline math spans and allow display equations to break columns.")
        plan = {
            "page_layout": {
                "format": "US Letter portrait",
                "columns": columns,
                "title_area": "Title spans top 30% with subtitle and metadata stacked",
                "margins": "1in margins, baseline grid for quick alignment",
            },
            "typography": {
                "title": "Large serif (e.g., 28pt) paired with thin rule",
                "headings": "Small caps with generous spacing",
                "body": "11pt serif with 14pt leading",
                "color_accents": "Muted blue for sidebars and figure captions",
            },
            "sections": sections,
            "callouts": callouts,
            "assets": assets,
            "global_guidelines": guidelines,
            "stats": {
                "word_count": word_count,
                "section_count": len(sections),
                "images_attached": len(images),
            },
            "narrative": (
                "Lead with a hero title, follow with overview prose, then transition into columnar content "
                "anchored by figures/tables called out in the draft."
            ),
        }
        return {"plan": plan, "warnings": []}

    def _split_sections(self, draft_text: str) -> List[Dict[str, Any]]:
        lines = draft_text.splitlines()
        sections: List[Dict[str, Any]] = []
        current_lines: List[str] = []
        current_title: str | None = None
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue
            if _looks_like_heading(stripped):
                if current_lines or current_title:
                    sections.append(self._build_section(current_title, current_lines))
                current_title = stripped
                current_lines = []
            else:
                current_lines.append(stripped)
        if current_lines or current_title:
            sections.append(self._build_section(current_title, current_lines))
        if not sections and draft_text:
            sections.append(self._build_section("Overview", draft_text.splitlines()))
        if not sections:
            sections.append(
                {
                    "id": "S1",
                    "title": "Layout Placeholder",
                    "goal": "Capture the expected structure once draft content is available.",
                    "layout": "Single column placeholder body text.",
                    "visuals": [],
                    "notes": [],
                }
            )
        for idx, section in enumerate(sections, start=1):
            section.setdefault("id", f"S{idx:02d}")
        return sections

    def _build_section(self, title: str | None, lines: Iterable[str]) -> Dict[str, Any]:
        text = " ".join(line.strip() for line in lines if line.strip())
        snippet = textwrap.shorten(text, width=220, placeholder="…") if text else "No content provided."
        title = title or "Body"
        visuals = []
        lower = text.lower()
        if any(keyword in lower for keyword in FIGURE_KEYWORDS):
            visuals.append("figure")
        if any(keyword in lower for keyword in TABLE_KEYWORDS):
            visuals.append("table")
        if "$" in text or "\\int" in text or "\\frac" in text:
            visuals.append("formula")
        layout = "Two-column narrative with inline callouts" if visuals else "Single-column body text"
        return {
            "id": None,
            "title": title.strip(),
            "goal": snippet,
            "layout": layout,
            "visuals": visuals,
            "notes": [],
        }

    def _collect_visual_requests(self, draft_text: str) -> List[Dict[str, str]]:
        lower = draft_text.lower()
        visuals: List[Dict[str, str]] = []
        for keyword in FIGURE_KEYWORDS:
            if keyword in lower:
                visuals.append({"type": "figure", "reason": f"Keyword '{keyword}' detected"})
                break
        for keyword in TABLE_KEYWORDS:
            if keyword in lower:
                visuals.append({"type": "table", "reason": f"Keyword '{keyword}' detected"})
                break
        math_hits = len(re.findall(r"(\\begin\{equation\}|\\\[|\$\$)", draft_text))
        if math_hits:
            visuals.append({"type": "formula", "reason": f"{math_hits} display math markers"})
        return visuals

    def _build_asset_slots(
        self, images: Sequence[Path], inferred_visuals: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        slots: List[Dict[str, Any]] = []
        for idx, path in enumerate(images, start=1):
            slots.append(
                {
                    "slot": f"asset-{idx:02d}",
                    "type": "figure",
                    "placement": "full-width introduction" if idx == 1 else "float",
                    "source": str(path),
                }
            )
        for idx, visual in enumerate(inferred_visuals, start=len(slots) + 1):
            slots.append(
                {
                    "slot": f"inferred-{idx:02d}",
                    "type": visual["type"],
                    "placement": "span-columns" if visual["type"] == "table" else "float",
                    "source": visual["reason"],
                }
            )
        return slots

    def _build_callouts(self, draft_text: str) -> List[Dict[str, Any]]:
        callouts: List[Dict[str, Any]] = []
        lower = draft_text.lower()
        for keyword in CALL_OUT_KEYWORDS:
            if keyword in lower:
                callouts.append(
                    {
                        "title": keyword.title(),
                        "placement": "right margin sidebar",
                        "purpose": "Highlight recurring tips or reminders mentioned in the draft.",
                    }
                )
                break
        if not callouts and draft_text:
            callouts.append(
                {
                    "title": "Key Findings",
                    "placement": "left sidebar",
                    "purpose": "Summarize the most important insight from the draft.",
                }
            )
        return callouts

    def _merge_plan(self, base_plan: Dict[str, Any], remote_plan: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base_plan)
        for key, value in remote_plan.items():
            merged[key] = value
        for key in ("page_layout", "typography", "sections", "callouts", "assets", "global_guidelines"):
            if key not in merged and key in base_plan:
                merged[key] = base_plan[key]
        if "stats" not in merged and "stats" in base_plan:
            merged["stats"] = base_plan["stats"]
        return merged

    def _enrich_plan(self, plan: Dict[str, Any], draft_text: str) -> None:
        flags = self._detect_content_flags(plan, draft_text)
        plan["content_flags"] = flags
        plan["chunks"] = plan.get("chunks") or self._build_chunk_outline(plan)
        doc_hint = self._infer_doc_class(plan, flags)
        if doc_hint:
            plan["doc_class_hint"] = doc_hint

    def _detect_content_flags(self, plan: Dict[str, Any], draft_text: str) -> Dict[str, bool]:
        sections = plan.get("sections") or []
        text = (draft_text or "").lower()
        visuals = [
            (vis or "").lower()
            for section in sections
            for vis in (section.get("visuals") or [])
        ]
        columns = str(plan.get("page_layout", {}).get("columns", "")).lower()
        return {
            "has_figures": any("figure" in v for v in visuals) or "figure" in text,
            "has_tables": any("table" in v for v in visuals) or "table" in text,
            "has_math": any("formula" in v or "math" in v for v in visuals)
            or any(tok in draft_text for tok in ("$", "\\int", "\\frac", "\\sum")),
            "has_code": "```" in draft_text or "code listing" in text,
            "has_appendix": "appendix" in text,
            "has_chapters": any(str(section.get("title", "")).lower().startswith("chapter") for section in sections),
            "two_column": "two" in columns,
        }

    def _infer_doc_class(self, plan: Dict[str, Any], flags: Dict[str, bool]) -> Dict[str, Any]:
        candidate = "lix_article"
        confidence = 0.6
        reason = "Defaulted to article layout."
        if flags.get("has_chapters"):
            candidate, confidence, reason = DOC_CLASS_RULES[0]
        elif flags.get("has_appendix") or len(plan.get("sections") or []) >= 6:
            candidate, confidence, reason = DOC_CLASS_RULES[1]
        elif flags.get("two_column"):
            candidate, confidence, reason = DOC_CLASS_RULES[2]
        return {
            "candidate": candidate,
            "confidence": confidence,
            "reason": reason,
            "flags": flags,
        }

    def _build_chunk_outline(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        outline: List[Dict[str, Any]] = []
        for idx, section in enumerate(plan.get("sections") or [], start=1):
            chunk_id = section.get("id") or f"S{idx:02d}"
            outline.append(
                {
                    "id": chunk_id,
                    "title": section.get("title") or f"Section {idx}",
                    "kind": self._chunk_kind(section),
                    "visuals": list(section.get("visuals") or []),
                    "layout": section.get("layout"),
                }
            )
        return outline

    def _chunk_kind(self, section: Dict[str, Any]) -> str:
        visuals = [str(v).lower() for v in (section.get("visuals") or [])]
        if any("figure" in v for v in visuals):
            return "figure"
        if any("table" in v for v in visuals):
            return "table"
        if any("formula" in v or "math" in v for v in visuals):
            return "math"
        return "section"


def _read_text_inputs(paths: Sequence[Path]) -> str:
    buf: List[str] = []
    for path in paths:
        if path.exists():
            buf.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(buf).strip()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Layout planner powered by Qwen2.5-VL (with heuristic fallback)")
    ap.add_argument("--text", type=str, default=None, help="Inline draft text")
    ap.add_argument("--text-file", action="append", type=Path, default=[], help="Path(s) to draft text files")
    ap.add_argument("--image", action="append", type=Path, default=[], help="Path(s) to draft images")
    ap.add_argument("--out", type=Path, default=Path("build/layout_plan.json"), help="JSON output path")
    ap.add_argument("--text-out", type=Path, default=None, help="Optional plaintext summary output")
    ap.add_argument("--model", type=str, default="Qwen2.5-VL-32B-Instruct", help="Model name served by vLLM")
    ap.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000/v1", help="OpenAI-compatible endpoint root")
    ap.add_argument("--api-key", type=str, default=None, help="API key for the endpoint (falls back to env)")
    ap.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    ap.add_argument("--max-tokens", type=int, default=900, help="Max tokens for the layout description")
    ap.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    ap.add_argument("--offline", action="store_true", help="Skip remote model call and use heuristics only")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    text = args.text or ""
    if args.text_file:
        text += ("\n\n" if text else "") + _read_text_inputs(args.text_file)
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    images = [p for p in args.image if p is not None]
    cfg = LayoutPlannerConfig(
        model=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        prefer_remote=not args.offline,
    )
    planner = LayoutPlanner(cfg)
    blueprint = planner.generate(text, images)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blueprint.to_json(), indent=2), encoding="utf-8")
    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(blueprint.as_text(), encoding="utf-8")
    print(f"[layout-planner] wrote {args.out}")
    if args.text_out:
        print(f"[layout-planner] wrote {args.text_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
