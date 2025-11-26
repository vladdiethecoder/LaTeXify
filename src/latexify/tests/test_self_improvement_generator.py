from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from latexify.self_improvement.generator import LLMPatchGenerator, GeneratorConfig
from latexify.self_improvement.models import AgentVersion, AgentMetrics
from latexify.self_improvement.knowledge_graph import KnowledgeGraph
from latexify.self_improvement.vector_memory import VectorMemory


def test_generator_returns_noop_when_dry_run():
    gen = LLMPatchGenerator(GeneratorConfig(dry_run=True))
    parent = AgentVersion(version_id="v0", parent_id=None, strategy="VALIDATION", metrics=AgentMetrics(score=0.5))
    proposals = gen.generate(parent, KnowledgeGraph(), VectorMemory())
    assert proposals
    assert proposals[0].operations == []


def test_generator_parses_json_payload():
    payload = """
    { "proposals": [{
        "candidate_id": "v1",
        "strategy": "TARGETED",
        "rationale": "fix bug",
        "target_tests": ["tests/test_smoke_release.py::test_smoke_pipeline_produces_rewards"],
        "ops": [{"file": "src/latexify/README.md", "search": "foo", "replace": "bar"}]
    }]}
    """

    def fake_text(prompt: str) -> str:
        return payload

    gen = LLMPatchGenerator(GeneratorConfig(dry_run=False), text_generator=fake_text)
    parent = AgentVersion(version_id="v0", parent_id=None, strategy="TARGETED", metrics=AgentMetrics(score=0.1))
    proposals = gen.generate(parent, KnowledgeGraph(), VectorMemory())
    assert proposals[0].candidate_id == "v1"
    assert proposals[0].target_tests == ["tests/test_smoke_release.py::test_smoke_pipeline_produces_rewards"]
    assert proposals[0].operations[0].file_path == Path("src/latexify/README.md")
