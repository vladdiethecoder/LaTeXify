from __future__ import annotations

"""Planner-aware specialist routing for deterministic snippet generation."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import synth_figure, synth_figure_placeholder, synth_formula, synth_table
from .synth_shared import SpecialistPrompt, load_specialist_prompt


FIGURE_HINTS = {"figure", "image", "picture", "graphic", "diagram", "photo", "chart"}
TABLE_HINTS = {"table", "spreadsheet", "tabular"}
MATH_HINTS = {"math", "formula", "equation", "expression"}

TAG_FIELDS = (
    "kind",
    "type",
    "content_type",
    "asset_source_type",
)


@dataclass(frozen=True)
class SpecialistDecision:
    name: str
    handler: Callable[[Dict], Tuple[str, List[str]]]
    reason: str
    prompt: SpecialistPrompt
    metadata: Dict[str, str]

    def run(self, bundle: Dict) -> Tuple[str, List[str]]:
        return self.handler(bundle)


def _normalize_tokens(value: str | None) -> Iterable[str]:
    if not value:
        return []
    parts = [p.strip().lower() for p in str(value).replace("-", " ").replace("_", " ").split() if p.strip()]
    return parts


def _collect_tags(bundle: Dict, plan_info: Dict[str, str]) -> List[str]:
    tags: List[str] = []
    for field in TAG_FIELDS:
        for source in (plan_info, bundle):
            raw = source.get(field) if isinstance(source, dict) else None
            for token in _normalize_tokens(raw):
                if token not in tags:
                    tags.append(token)
    return tags


def _match_any(tags: Iterable[str], candidates: set[str]) -> bool:
    return any(tag in candidates for tag in tags)


def _route_for_tags(tags: List[str]) -> Optional[Tuple[str, Callable[[Dict], Tuple[str, List[str]]]]]:
    if "figure" in tags and "placeholder" in tags:
        return "figure_placeholder", synth_figure_placeholder.synthesize
    if "placeholder" in tags:
        return "figure_placeholder", synth_figure_placeholder.synthesize
    if _match_any(tags, FIGURE_HINTS):
        return "figure", synth_figure.synthesize
    if _match_any(tags, TABLE_HINTS):
        return "table", synth_table.synthesize
    if _match_any(tags, MATH_HINTS):
        return "math", synth_formula.synthesize
    return None


def select_specialist(bundle: Dict, plan_info: Dict[str, str]) -> Optional[SpecialistDecision]:
    prompt = load_specialist_prompt()
    tags = _collect_tags(bundle, plan_info)
    route = _route_for_tags(tags)
    if not route:
        if plan_info.get("kind") == "figure_placeholder":
            route = ("figure_placeholder", synth_figure_placeholder.synthesize)
        else:
            return None
    name, handler = route
    metadata = {
        "prompt_version": prompt.version,
        "prompt_path": "tasks/synthesis_agent.md",
        "specialist": name,
        "tags": ",".join(tags),
    }
    reason = f"specialist:{name}"
    return SpecialistDecision(name=name, handler=handler, reason=reason, prompt=prompt, metadata=metadata)
