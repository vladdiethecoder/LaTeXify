from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .models import AgentVersion
from .patching import PatchProposal, PatchOperation
from .knowledge_graph import KnowledgeGraph
from .vector_memory import VectorMemory

LOGGER = logging.getLogger(__name__)

TextGenerator = Callable[[str], str]


@dataclass
class GeneratorConfig:
    model_path: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    device: str = "auto"
    use_vllm: bool = True
    temperature: float = 0.4
    max_candidates: int = 1
    dry_run: bool = True  # If True, we return no-op patches for safety.


class LLMPatchGenerator:
    """
    LLM-driven patch generator with multi-strategy prompts and multi-candidate output.

    The generator expects the LLM to return JSON with fields:
    {
      "proposals": [
        {
          "candidate_id": "...",
          "strategy": "TARGETED",
          "rationale": "...",
          "target_tests": ["tests/test_smoke_release.py::test_smoke_pipeline_produces_rewards"],
          "ops": [
            {"file": "path/relative.py", "search": "old", "replace": "new"}
          ]
        }
      ]
    }
    """

    def __init__(self, config: GeneratorConfig, text_generator: Optional[TextGenerator] = None, log_path: Optional[Path] = None):
        self.config = config
        self.text_generator = text_generator
        self.log_path = log_path or Path("logs/llm_raw_outputs.txt")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, parent: AgentVersion, graph: KnowledgeGraph, memory: VectorMemory) -> List[PatchProposal]:
        # Safety: if no text generator or dry_run, return no-op proposals so the loop can still proceed.
        if self.config.dry_run or self.text_generator is None:
            return [
                PatchProposal(
                    candidate_id=f"{parent.version_id}-noop",
                    strategy="VALIDATION",
                    rationale="Dry-run noop patch; supply text_generator to enable real edits.",
                    operations=[],
                    target_tests=list(parent.metrics.failed_tasks.keys()),
                )
            ]

        similar = memory.query(parent.summary or "", k=3)
        prompt = self._build_prompt(parent, graph, similar)
        raw = self.text_generator(prompt)
        return self._parse_response(raw, parent)

    def _build_prompt(self, parent: AgentVersion, graph: KnowledgeGraph, similar) -> str:
        edges = graph.find_edges(src=parent.version_id)
        lines = "\n".join(f"- {e['relation']} -> {e['dst']}" for e in edges)
        similar_txt = "\n".join(f"- {k}: {score:.2f}" for k, score in similar)
        failing = list(parent.metrics.failed_tasks.keys())
        return (
            "You are a self-improving coding agent. "
            "Select a strategy (TARGETED, EXPLORATORY, SYNTHESIS, VALIDATION, CREATIVE, CRITICAL) and propose up to "
            f"{self.config.max_candidates} candidate patches.\n"
            "Return ONLY JSON wrapped in ```json ... ``` with this exact schema:\n"
            "```json\n"
            "{\n"
            "  \"proposals\": [\n"
            "    {\n"
            "      \"candidate_id\": \"v1\",\n"
            "      \"strategy\": \"TARGETED\",\n"
            "      \"rationale\": \"why this helps\",\n"
            "      \"target_tests\": [\"src/latexify/tests/test_smoke_release.py::test_smoke_pipeline_produces_rewards\"],\n"
            "      \"ops\": [\n"
            "        {\"file\": \"path/relative.py\", \"search\": \"old text\", \"replace\": \"new text\"}\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "If unsure, return a single proposal with empty ops and the failing tests in target_tests.\n"
            f"Parent: {parent.version_id}, score={parent.metrics.score}\n"
            f"Known failures: {failing}\n"
            f"Lineage: \n{lines}\n"
            f"Similar memories: \n{similar_txt}\n"
        )

    def _parse_response(self, raw: str, parent: AgentVersion) -> List[PatchProposal]:
        blob = self._extract_json(raw)
        if not blob:
            LOGGER.warning("Failed to locate JSON in LLM output. Falling back to noop.")
            self._log_raw(raw)
            return self._fallback(parent)
        try:
            repaired = self._balance_json(blob)
            data = json.loads(repaired)
        except Exception as exc:
            LOGGER.warning("Failed to parse LLM JSON (%s). Raw snippet: %s", exc, blob[:2000])
            self._log_raw(raw)
            return self._fallback(parent)

        proposals: List[PatchProposal] = []
        for item in data.get("proposals", []):
            ops: List[PatchOperation] = []
            for op in item.get("ops", []):
                try:
                    ops.append(
                        PatchOperation(
                            file_path=Path(op["file"]),
                            search=op["search"],
                            replace=op["replace"],
                        )
                    )
                except KeyError:
                    continue
            proposals.append(
                PatchProposal(
                    candidate_id=item.get("candidate_id", f"{parent.version_id}-cand"),
                    strategy=item.get("strategy", "TARGETED"),
                    rationale=item.get("rationale", ""),
                    operations=ops,
                    target_tests=item.get("target_tests", list(parent.metrics.failed_tasks.keys())),
                )
            )
        if not proposals:
            return self._fallback(parent)
        return proposals

    def _fallback(self, parent: AgentVersion) -> List[PatchProposal]:
        return [
            PatchProposal(
                candidate_id=f"{parent.version_id}-fallback",
                strategy="VALIDATION",
                rationale="Failed to parse LLM output; noop.",
                operations=[],
                target_tests=list(parent.metrics.failed_tasks.keys()),
            )
        ]

    def _extract_json(self, raw: str) -> Optional[str]:
        """
        Heuristic JSON extraction: prefer fenced ```json blocks, else the smallest
        brace block containing the word "proposals".
        """
        # Regex capture of fenced JSON
        fence_all = re.findall(r"```json\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        if fence_all:
            return fence_all[-1].strip()

        # Generic fenced block if model omitted 'json' hint
        fence_plain = re.findall(r"```\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence_plain:
            return fence_plain[-1].strip()

        # Fenced block
        if "```" in raw:
            parts = raw.split("```")
            for i in range(len(parts) - 1):
                # Case 1: ```json\n{...}\n```
                if parts[i].strip().lower().startswith("json"):
                    segment = parts[i].strip()
                    segment = segment[4:].lstrip()  # drop 'json'
                    return segment
                # Case 2: ```json\n ... \n```
                if "json" in parts[i].lower():
                    segment = parts[i + 1].strip()
                    if segment.lower().startswith("json"):
                        segment = segment[4:].lstrip()
                    return segment
        # Find block containing "proposals"
        key_idx = raw.find("proposals")
        if key_idx == -1:
            return None
        # Walk backward to nearest '{'
        start = raw.rfind("{", 0, key_idx)
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return None
        # Try to balance braces from start
        depth = 0
        for j in range(start, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : j + 1]
        if start != -1 and end != -1:
            return raw[start : end + 1]
        # As last resort, grab first balanced braces
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            return raw[first : last + 1]
        return None

    def _log_raw(self, raw: str) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(raw)
                f.write("\n---\n")
        except Exception:
            pass

    def _balance_json(self, text: str) -> str:
        """
        Naive brace/bracket balancer to close truncated JSON.
        """
        open_curly = 0
        open_square = 0
        for ch in text:
            if ch == "{":
                open_curly += 1
            elif ch == "}":
                open_curly = max(0, open_curly - 1)
            elif ch == "[":
                open_square += 1
            elif ch == "]":
                open_square = max(0, open_square - 1)
        return text + ("]" * open_square) + ("}" * open_curly)
