"""Utilities for loading curated reference_tex exemplars."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference_tex"
DISABLE_TOKENS = {"none", "off", "disable", "disabled"}


def load_reference_tex(
    domain: str | None,
    *,
    max_files: int = 3,
    max_chars: int = 1800,
    root: Path | None = None,
) -> List[str]:
    """Load curated LaTeX exemplars for the requested domain.

    Args:
        domain: Subdirectory label under release/reference_tex/.
        max_files: Cap on the number of files to read (to bound prompt size).
        max_chars: Truncate each file to this many characters.
        root: Optional override for the reference_tex root (useful in tests).
    """

    if max_files <= 0 or max_chars <= 0:
        return []
    normalized = (domain or "default").strip()
    if normalized.lower() in DISABLE_TOKENS:
        LOGGER.info("[refiner] Style exemplars disabled via --style-domain=%s", domain)
        return []

    search_roots = []
    root_dir = root or REFERENCE_ROOT
    if normalized:
        search_roots.append((normalized, root_dir / normalized))
    if normalized.lower() != "default":
        search_roots.append(("default", root_dir / "default"))

    seen_domains = set()
    for domain_label, domain_dir in search_roots:
        if domain_label in seen_domains:
            continue
        seen_domains.add(domain_label)
        if not domain_dir.exists():
            LOGGER.debug("reference_tex domain %s missing at %s", domain_label, domain_dir)
            continue
        tex_files = sorted(domain_dir.glob("*.tex"))
        if not tex_files:
            LOGGER.debug("reference_tex domain %s has no .tex files", domain_label)
            continue
        exemplars: List[str] = []
        for tex_path in tex_files:
            try:
                text = tex_path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError as exc:  # pragma: no cover - filesystem issues
                LOGGER.warning("Failed to read exemplar %s (%s)", tex_path, exc)
                continue
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "\n% ... exemplar truncated ..."
            exemplars.append(text)
            if len(exemplars) >= max_files:
                break
        if exemplars:
            LOGGER.info(
                "[refiner] Loaded %s reference_tex exemplars for domain '%s'",
                len(exemplars),
                domain_label,
            )
            return exemplars
    LOGGER.warning(
        "[refiner] No reference_tex exemplars found for domain '%s'; running in generic mode",
        normalized,
    )
    return []


__all__ = ["load_reference_tex"]
