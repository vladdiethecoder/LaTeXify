"""
Self-improvement scaffolding inspired by DGM and ITRS.

This package provides lightweight primitives (archive, patch ops, logging,
knowledge graph stubs) so the LaTeXify agents can run iterative
select → propose → apply → validate loops.
"""

from .models import AgentMetrics, AgentVersion, StrategyChoice, EvolutionConfig
from .archive import Archive
from .thought_log import ThoughtLog
from .knowledge_graph import KnowledgeGraph
from .vector_memory import VectorMemory
from .patching import PatchOperation, PatchProposal, apply_patch_proposal
from .validator import PatchValidator, ValidationResult
from .loop import EvolutionRunner
from .report import render_report
from .generator import LLMPatchGenerator, GeneratorConfig
from .evaluator import EvaluatorRunner, EvaluationConfig
from .text_llm import LocalTextGenerator, LocalLLMConfig, GGUFTextGenerator, GGUFLLMConfig

__all__ = [
    "AgentMetrics",
    "AgentVersion",
    "StrategyChoice",
    "EvolutionConfig",
    "Archive",
    "ThoughtLog",
    "KnowledgeGraph",
    "VectorMemory",
    "PatchOperation",
    "PatchProposal",
    "apply_patch_proposal",
    "PatchValidator",
    "ValidationResult",
    "EvolutionRunner",
    "render_report",
    "LLMPatchGenerator",
    "GeneratorConfig",
    "EvaluatorRunner",
    "EvaluationConfig",
    "LocalTextGenerator",
    "LocalLLMConfig",
    "GGUFTextGenerator",
    "GGUFLLMConfig",
]
