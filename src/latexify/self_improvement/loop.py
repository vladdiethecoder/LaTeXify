from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .archive import Archive
from .knowledge_graph import KnowledgeGraph
from .models import AgentVersion, EvolutionConfig, StrategyChoice
from .patching import PatchProposal, apply_patch_proposal
from .thought_log import ThoughtLog, ThoughtEntry
from .validator import PatchValidator
from .vector_memory import VectorMemory

LOGGER = logging.getLogger(__name__)

# Types for pluggable behaviors
PatchGenerator = Callable[[AgentVersion, KnowledgeGraph, VectorMemory], List[PatchProposal]]
Evaluator = Callable[[AgentVersion], AgentVersion]


class EvolutionRunner:
    """
    Orchestrates the DGM-style loop: select → propose → apply → validate → archive.
    """

    def __init__(
        self,
        repo_root: Path,
        archive: Archive,
        config: EvolutionConfig,
        patch_generator: PatchGenerator,
        evaluator: Evaluator,
        thought_log: Optional[ThoughtLog] = None,
        graph: Optional[KnowledgeGraph] = None,
        vector_memory: Optional[VectorMemory] = None,
    ):
        self.repo_root = repo_root
        self.archive = archive
        self.config = config
        self.patch_generator = patch_generator
        self.evaluator = evaluator
        self.graph = graph or KnowledgeGraph()
        self.vector_memory = vector_memory or VectorMemory()
        self.thought_log = thought_log or ThoughtLog(config.thought_log_path)
        self.validator = PatchValidator(repo_root)

    def run(self) -> Archive:
        for iteration in range(1, self.config.max_generations + 1):
            parent = self.archive.select_parent(temperature=self.config.exploration_temperature)
            if not parent:
                LOGGER.warning("Archive empty; stopping evolution.")
                break

            LOGGER.info("Iteration %s: selected parent %s", iteration, parent.version_id)
            proposals = self.patch_generator(parent, self.graph, self.vector_memory)
            if not proposals:
                LOGGER.info("No proposals returned; stopping.")
                break

            accepted_child: Optional[AgentVersion] = None
            for proposal in proposals[: self.config.allow_parallel_candidates]:
                LOGGER.info("Applying proposal %s (%s)", proposal.candidate_id, proposal.strategy)
                result = apply_patch_proposal(proposal, self.repo_root)
                if not result.success:
                    LOGGER.warning("Patch application failed: %s", result.errors)
                    continue

                target_tests = proposal.target_tests or list(parent.metrics.failed_tasks.keys())
                validated = self.validator.run_pytest(target_tests)
                child = self.evaluator(
                    AgentVersion(
                        version_id=proposal.candidate_id,
                        parent_id=parent.version_id,
                        strategy=proposal.strategy,  # type: ignore[arg-type]
                        summary=proposal.rationale,
                    )
                )

                regression = parent.metrics.score - child.metrics.score
                keep_child = child.metrics.score >= parent.metrics.score or (
                    self.config.retain_neutral and regression <= self.config.regression_tolerance
                )
                if keep_child:
                    LOGGER.info("Accepted child %s score=%.3f (parent %.3f)", child.version_id, child.metrics.score, parent.metrics.score)
                    self._record(iteration, child, proposal, validated.output, accepted=True)
                    self._update_memories(child, proposal)
                    self.archive.add(child)
                    accepted_child = child
                    break
                else:
                    self._record(iteration, child, proposal, validated.output, accepted=False)

            self.archive.maybe_prune(self.config.archive_limit)

            if accepted_child is None:
                LOGGER.info("No child accepted in iteration %s; stopping early.", iteration)
                break
        return self.archive

    def _record(self, iteration: int, child: AgentVersion, proposal: PatchProposal, validation_output: str, accepted: bool) -> None:
        entry = ThoughtEntry(
            iteration=iteration,
            agent_id=child.version_id,
            strategy=proposal.strategy,
            summary=proposal.rationale,
            details={
                "accepted": accepted,
                "parent": child.parent_id,
                "score": child.metrics.score,
                "validation": validation_output[:2000],
            },
        )
        self.thought_log.append(entry)
        self.graph.add_node(child.version_id, {"score": child.metrics.score, "accepted": accepted})
        if child.parent_id:
            self.graph.add_edge(child.parent_id, "child", child.version_id, {"strategy": proposal.strategy})

        for task in child.metrics.passed_tasks:
            self.graph.add_edge(child.version_id, "passed", task)
        for task, err in child.metrics.failed_tasks.items():
            self.graph.add_edge(child.version_id, "failed", task, {"error": err})

    def _update_memories(self, child: AgentVersion, proposal: PatchProposal) -> None:
        text = f"{child.version_id} {proposal.strategy} {proposal.rationale} score={child.metrics.score}"
        self.vector_memory.add(child.version_id, text)
