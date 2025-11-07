from __future__ import annotations

"""Planner-aware specialist routing for deterministic snippet generation."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import synth_figure, synth_figure_placeholder, synth_formula, synth_table, synth_text
from .model_backends import LlamaCppBackend, LlamaCppConfig
from .synth_shared import SpecialistPrompt, load_specialist_prompt
from ..utils.paths import tasks_root


_ROUTER_PROMPT_CACHE: Optional[SpecialistPrompt] = None


def _extract_prompt_version(text: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip() or "unknown"
    for line in text.splitlines():
        if line.startswith("#"):
            return line.strip("# ") or "unknown"
    return "unknown"


def load_router_prompt(root: Path | None = None) -> SpecialistPrompt:
    global _ROUTER_PROMPT_CACHE
    if _ROUTER_PROMPT_CACHE is not None:
        return _ROUTER_PROMPT_CACHE
    if root is None:
        prompt_root = tasks_root()
    else:
        prompt_root = Path(root)
    prompt_path = prompt_root / "router.md"
    body = prompt_path.read_text(encoding="utf-8")
    prompt = SpecialistPrompt(version=_extract_prompt_version(body), body=body)
    _ROUTER_PROMPT_CACHE = prompt
    return prompt


@dataclass(frozen=True)
class SpecialistDecision:
    name: str
    handler: Callable[[Dict], Tuple[str, List[str]]]
    reason: str
    prompt: SpecialistPrompt
    metadata: Dict[str, str]

    def run(self, bundle: Dict) -> Tuple[str, List[str]]:
        return self.handler(bundle)


class SpecialistRouter:
    """Prompt-driven specialist selection with model-backed classification."""

    FIGURE_HINTS = {"figure", "image", "picture", "graphic", "diagram", "photo", "chart"}
    TABLE_HINTS = {"table", "spreadsheet", "tabular"}
    MATH_HINTS = {"math", "formula", "equation", "expression"}
    CODE_HINTS = {"code", "program", "snippet", "pseudo", "algorithm"}
    TAG_FIELDS = ("kind", "type", "content_type", "asset_source_type")
    VALID_TAGS = {"text", "table", "math", "figure", "figure_placeholder", "code"}
    TAG_SYNONYMS = {
        "image": "figure",
        "graphic": "figure",
        "diagram": "figure",
        "picture": "figure",
        "chart": "figure",
        "placeholder": "figure_placeholder",
        "figure-placeholder": "figure_placeholder",
        "equation": "math",
        "expression": "math",
        "formula": "math",
        "spreadsheet": "table",
        "tabular": "table",
        "program": "code",
        "snippet": "code",
        "pseudo": "code",
        "algorithm": "code",
    }
    ROUTER_MODEL_ENV = "LATEXIFY_ROUTER_MODEL"

    def __init__(
        self,
        plan: Dict | None = None,
        *,
        model_client: Optional[Callable[[str], str]] = None,
    ):
        self._synthesis_prompt = load_specialist_prompt()
        self._router_prompt = load_router_prompt()
        self._plan_lookup: Dict[str, Dict[str, str]] = {}
        if isinstance(plan, dict):
            for task in plan.get("tasks", []):
                if not isinstance(task, dict):
                    continue
                task_id = task.get("id")
                if task_id is None:
                    continue
                self._plan_lookup[str(task_id)] = self._normalize_metadata(task)
        self._model_client, self._router_model_name = self._resolve_model_client(model_client)
        self._last_classification_source = "heuristic"
        self._last_raw_response = ""

    def classify(self, task_bundle: Dict) -> str:
        payload = json.dumps(task_bundle, ensure_ascii=False, indent=2, sort_keys=True)
        prompt = "{}\n\n<task_bundle>\n{}\n</task_bundle>\n".format(
            self._router_prompt.body.strip(),
            payload,
        )
        raw_response = ""
        tag: Optional[str] = None
        if self._model_client is not None:
            try:
                raw_response = self._model_client(prompt) or ""
                tag = self._normalize_tag(raw_response)
            except Exception:
                raw_response = ""
                tag = None
        if not tag:
            bundle = task_bundle.get("bundle", {}) if isinstance(task_bundle, dict) else {}
            plan = task_bundle.get("plan", {}) if isinstance(task_bundle, dict) else {}
            tags = task_bundle.get("tags") if isinstance(task_bundle, dict) else None
            tag = self._heuristic_tag(bundle, plan, tags)
            self._last_classification_source = "heuristic"
        else:
            self._last_classification_source = "model"
        self._last_raw_response = raw_response.strip()
        return tag or "text"

    def route(self, bundle: Dict, plan_info: Dict | None = None) -> SpecialistDecision:
        task_id = str(bundle.get("task_id") or bundle.get("id") or "")
        plan_meta: Dict[str, str] = {}
        if plan_info:
            plan_meta.update(self._normalize_metadata(plan_info))
        if task_id and task_id in self._plan_lookup:
            plan_meta = {**self._plan_lookup[task_id], **plan_meta}
        tags = self._collect_tags(bundle, plan_meta)
        router_payload = {"bundle": bundle, "plan": plan_meta, "tags": tags}
        tag = self.classify(router_payload)
        name, handler = self._resolve_specialist(tag)
        metadata = {
            "prompt_version": self._synthesis_prompt.version,
            "prompt_path": "docs/tasks/synthesis_agent.md",
            "specialist": name,
            "tags": ",".join(tags),
            "router_prompt_version": self._router_prompt.version,
            "router_prompt_path": "docs/tasks/router.md",
            "router_model": self._router_model_name,
            "router_source": self._last_classification_source,
        }
        if self._last_raw_response:
            metadata["router_raw_response"] = self._last_raw_response
        reason = f"router:{self._last_classification_source}:{name}"
        return SpecialistDecision(
            name=name,
            handler=handler,
            reason=reason,
            prompt=self._synthesis_prompt,
            metadata=metadata,
        )

    def _resolve_model_client(
        self,
        override: Optional[Callable[[str], str]] | Optional[Tuple[Callable[[str], str], str]],
    ) -> Tuple[Optional[Callable[[str], str]], str]:
        if isinstance(override, tuple):
            client, name = override
            return client, name
        if callable(override):
            return override, "override"
        env_path = os.environ.get(self.ROUTER_MODEL_ENV)
        if not env_path:
            return None, "heuristic"
        model_path = Path(env_path).expanduser()
        if not model_path.exists():
            return None, "heuristic"
        cfg = LlamaCppConfig(
            model_path=model_path,
            n_ctx=2048,
            n_batch=256,
            seed=1337,
            tensor_split="auto",
            verbose=False,
        )
        backend = LlamaCppBackend(cfg)

        def _client(prompt: str) -> str:
            return backend.generate(
                prompt,
                max_tokens=8,
                temperature=0.0,
                top_p=0.1,
                top_k=20,
                stop=["\n"],
                repeat_penalty=1.0,
            )

        return _client, model_path.name or str(model_path)

    def _heuristic_tag(
        self,
        bundle: Dict,
        plan_info: Dict,
        tags: Optional[Iterable[str]],
    ) -> str:
        tag_list = list(tags) if tags is not None else self._collect_tags(bundle, plan_info)
        normalized_tags = [self._normalize_tag(tag) for tag in tag_list]
        raw_tags = [str(tag).lower() for tag in tag_list if isinstance(tag, str)]
        if "figure_placeholder" in normalized_tags:
            return "figure_placeholder"
        if plan_info.get("kind") == "figure_placeholder":
            return "figure_placeholder"
        if any(tag in ("figure", "figure_placeholder") for tag in normalized_tags):
            return "figure"
        if any(tag in self.FIGURE_HINTS for tag in raw_tags):
            return "figure"
        if any(tag in ("table",) for tag in normalized_tags):
            return "table"
        if any(tag in self.TABLE_HINTS for tag in raw_tags):
            return "table"
        if any(tag in ("math",) for tag in normalized_tags):
            return "math"
        if any(tag in self.MATH_HINTS for tag in raw_tags):
            return "math"
        if any(tag in ("code",) for tag in normalized_tags):
            return "code"
        if any(tag in self.CODE_HINTS for tag in raw_tags):
            return "code"
        return "text"

    def _resolve_specialist(
        self,
        tag: str,
    ) -> Tuple[str, Callable[[Dict], Tuple[str, List[str]]]]:
        normalized = self._normalize_tag(tag) or "text"
        if normalized == "figure_placeholder":
            return "figure_placeholder", synth_figure_placeholder.synthesize
        if normalized == "figure":
            return "figure", synth_figure.synthesize
        if normalized == "table":
            return "table", synth_table.synthesize
        if normalized == "math":
            return "math", synth_formula.synthesize
        # Treat code as a text specialist for now.
        return "text", synth_text.synthesize

    @classmethod
    def _collect_tags(cls, bundle: Dict, plan_info: Dict[str, str]) -> List[str]:
        tags: List[str] = []
        for field in cls.TAG_FIELDS:
            for source in (plan_info, bundle):
                raw = source.get(field) if isinstance(source, dict) else None
                for token in cls._normalize_tokens(raw):
                    if token not in tags:
                        tags.append(token)
        return tags

    @staticmethod
    def _normalize_tokens(value: str | None) -> Iterable[str]:
        if not value:
            return []
        parts = [
            p.strip().lower()
            for p in str(value).replace("-", " ").replace("_", " ").split()
            if p.strip()
        ]
        return parts

    @classmethod
    def _normalize_tag(cls, value: str | None) -> Optional[str]:
        if not value:
            return None
        token = str(value).strip().lower()
        if not token:
            return None
        token = token.replace("\n", " ").replace("\r", " ")
        token = token.split()[0]
        token = token.strip().strip(".,:;!?")
        token = token.replace("-", "-")
        token = cls.TAG_SYNONYMS.get(token, token)
        if token in cls.VALID_TAGS:
            return token
        return None

    @staticmethod
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
                for sub_key in ("kind", "type", "content_type", "asset_source_type"):
                    sub_value = value.get(sub_key)
                    if sub_value is not None and sub_key not in normalized:
                        normalized[sub_key] = str(sub_value)
        return normalized
