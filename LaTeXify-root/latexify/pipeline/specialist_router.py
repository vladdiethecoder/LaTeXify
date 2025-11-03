from __future__ import annotations

"""Planner-aware specialist routing for deterministic snippet generation."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import synth_figure, synth_figure_placeholder, synth_formula, synth_table, synth_text
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


def _normalize_metadata(source: Dict | None) -> Dict[str, str]:
    if not isinstance(source, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in source.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float)):
            normalized[str(key)] = str(value)
        elif isinstance(value, dict):
            # Flatten only selected nested hints
            for sub_key in ("kind", "type", "content_type", "asset_source_type"):
                sub_value = value.get(sub_key)
                if sub_value is not None:
                    normalized.setdefault(sub_key, str(sub_value))
    return normalized


class SpecialistRouter:
    """Light-weight wrapper that caches prompt + plan metadata for routing."""

    def __init__(self, plan: Dict | None = None):
        self._prompt = load_specialist_prompt()
        self._plan_lookup: Dict[str, Dict[str, str]] = {}
        if isinstance(plan, dict):
            for task in plan.get("tasks", []):
                if not isinstance(task, dict):
                    continue
                task_id = task.get("id")
                if task_id is None:
                    continue
                self._plan_lookup[str(task_id)] = _normalize_metadata(task)

    @property
    def prompt(self) -> SpecialistPrompt:
        return self._prompt

    def classify(self, bundle: Dict, plan_task: Dict | None = None) -> SpecialistDecision:
        task_id = str(bundle.get("task_id") or bundle.get("id") or "")
        plan_info: Dict[str, str] = {}
        if plan_task:
            plan_info.update(_normalize_metadata(plan_task))
        if task_id in self._plan_lookup:
            plan_info.update(self._plan_lookup[task_id])
        decision = select_specialist(bundle, plan_info)
        if decision:
            return decision
        metadata = {
            "prompt_version": self._prompt.version,
            "prompt_path": "tasks/synthesis_agent.md",
            "specialist": "text",
            "tags": ",".join(_collect_tags(bundle, plan_info)),
        }
        reason = "specialist:text"
        return SpecialistDecision(
            name="text",
            handler=synth_text.synthesize,
            reason=reason,
            prompt=self._prompt,
            metadata=metadata,
        )
