"""Aesthetic LaTeX image rendering utilities."""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont

from ..core import common

LOGGER = logging.getLogger(__name__)


@dataclass
class LaTeXImageStyle:
    name: str
    document_class: str
    font: str | None
    font_size: int
    margin: int
    background: tuple[int, int, int]
    accent: tuple[int, int, int]


class LaTeXImageGenerator:
    def __init__(self, styles: Sequence[LaTeXImageStyle] | None = None, max_regenerations: int = 3) -> None:
        self.styles = list(styles) if styles else self._default_styles()
        self.max_regenerations = max(1, max_regenerations)

    def generate(self, chunks: Sequence[common.Chunk], output_dir: Path) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"requested": len(chunks), "rendered": 0, "fallback": 0}
        failures: List[str] = []
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata or {})
            try:
                image_path, meta = self._render_with_retries(chunk, output_dir)
            except Exception as exc:  # pragma: no cover - best effort rendering
                LOGGER.debug("latex_image_generation failed for %s: %s", chunk.chunk_id, exc)
                failures.append(chunk.chunk_id)
                continue
            if image_path:
                chunk.metadata["latex_image"] = str(image_path)
                chunk.metadata["latex_image_style"] = meta.get("style")
                chunk.metadata["latex_image_attempts"] = meta.get("attempts")
                metrics["rendered"] += 1
            else:
                metrics["fallback"] += 1
        metrics["failures"] = len(failures)
        return {"metrics": metrics, "failures": failures}

    # Internal helpers -----------------------------------------------------

    def _render_with_retries(
        self,
        chunk: common.Chunk,
        output_dir: Path,
    ) -> tuple[str | None, Dict[str, object]]:
        for attempt in range(1, self.max_regenerations + 1):
            style = self.styles[min(attempt - 1, len(self.styles) - 1)]
            try:
                image_path = self._compose_image(chunk, output_dir, style)
                return image_path, {"style": style.name, "attempts": attempt}
            except Exception as exc:
                LOGGER.debug("Style %s failed for %s: %s", style.name, chunk.chunk_id, exc)
                continue
        try:
            fallback_path = self._basic_render(chunk, output_dir)
            return fallback_path, {"style": "basic", "attempts": self.max_regenerations + 1}
        except Exception as exc:  # pragma: no cover - IO errors
            LOGGER.debug("Basic render failed for %s: %s", chunk.chunk_id, exc)
            branch_path = self._write_branch_a_fallback(chunk, output_dir)
            return branch_path, {"style": "branch_a"}

    def _compose_image(self, chunk: common.Chunk, output_dir: Path, style: LaTeXImageStyle) -> str:
        metadata = chunk.metadata or {}
        region = metadata.get("region_type", "text")
        width, height = self._estimate_canvas(chunk.text, region, style.margin)
        image = Image.new("RGB", (width, height), color=style.background)
        draw = ImageDraw.Draw(image)
        font = self._resolve_font(style.font, style.font_size)
        header = f"\\documentclass{{{style.document_class}}}"
        draw.text((style.margin, style.margin // 2), header, font=font, fill=style.accent)
        body_top = style.margin + style.font_size + 10
        wrapped = self._wrap_text(chunk.text or "", width - (2 * style.margin), font)
        draw.multiline_text((style.margin, body_top), wrapped, font=font, fill=(20, 20, 20), spacing=6)
        target = output_dir / f"{chunk.chunk_id}_style_{style.name}.png"
        image.save(target, "PNG")
        return str(target)

    def _basic_render(self, chunk: common.Chunk, output_dir: Path) -> str:
        fallback_dir = output_dir / "basic"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (800, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = self._resolve_font(None, 20)
        wrapped = self._wrap_text(chunk.text or "", 760, font)
        draw.multiline_text((20, 20), wrapped, font=font, fill=(0, 0, 0))
        target = fallback_dir / f"{chunk.chunk_id}.png"
        image.save(target, "PNG")
        chunk.metadata["latex_image_fallback"] = str(target)
        return str(target)

    def _write_branch_a_fallback(self, chunk: common.Chunk, output_dir: Path) -> str:
        branch_dir = output_dir / "branch_a_fallback"
        branch_dir.mkdir(parents=True, exist_ok=True)
        target = branch_dir / f"{chunk.chunk_id}.txt"
        target.write_text(chunk.text or "", encoding="utf-8")
        chunk.metadata["latex_image_fallback"] = str(target)
        return None

    def _estimate_canvas(self, text: str, region: str, margin: int) -> tuple[int, int]:
        lines = max(1, text.count("\n") + 1)
        base_width = 1200 if region != "equation" else 1000
        base_height = 400 + (lines * 18)
        if region in {"figure", "table"}:
            base_width = 1600
            base_height = max(base_height, 900)
        width = base_width + (margin * 2)
        height = base_height + margin
        return width, height

    def _resolve_font(self, font_name: str | None, size: int) -> ImageFont.ImageFont:
        if font_name:
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:  # pragma: no cover - font missing
                LOGGER.debug("Font %s unavailable; falling back to default.", font_name)
        try:
            return ImageFont.truetype("DejaVuSerif.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_text(self, text: str, max_width: int, font: ImageFont.ImageFont) -> str:
        if not text:
            return "(empty chunk)"
        avg_char = font.getlength("M") or 10
        chars_per_line = max(20, int(max_width / max(1.0, avg_char)))
        wrapper = textwrap.TextWrapper(width=chars_per_line)
        lines = []
        for paragraph in text.splitlines():
            stripped = paragraph.strip()
            if not stripped:
                lines.append("")
                continue
            lines.extend(wrapper.wrap(stripped))
        return "\n".join(lines)

    @staticmethod
    def _default_styles() -> List[LaTeXImageStyle]:
        return [
            LaTeXImageStyle(
                name="journal",
                document_class="article",
                font="DejaVuSerif.ttf",
                font_size=28,
                margin=80,
                background=(249, 247, 243),
                accent=(120, 94, 62),
            ),
            LaTeXImageStyle(
                name="memoir",
                document_class="memoir",
                font="DejaVuSans.ttf",
                font_size=26,
                margin=70,
                background=(245, 245, 248),
                accent=(70, 70, 90),
            ),
            LaTeXImageStyle(
                name="minimal",
                document_class="article",
                font=None,
                font_size=22,
                margin=60,
                background=(255, 255, 255),
                accent=(0, 0, 0),
            ),
        ]


__all__ = ["LaTeXImageGenerator", "LaTeXImageStyle"]
