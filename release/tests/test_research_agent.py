from release.agents.research_agent import ResearchAgent, ResearchSnippet
from release.agents.graph_state import GraphState


class FakeBackend:
    def fetch(self, query, max_results):
        return [ResearchSnippet(source="fake", content=f"content:{query}", url="https://example.com")]


def test_research_agent_custom_backend():
    agent = ResearchAgent(backend=FakeBackend(), max_results=1)
    state = GraphState(chunk_id="c1", content="test content")

    updated = agent.augment(state)
    assert updated.research_snippets
    assert "example.com" in updated.research_snippets[0]


def test_research_agent_offline_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_AGENT_OFFLINE", "1")
    agent = ResearchAgent(max_results=1)
    snippets = agent.search("layout hints")
    assert snippets and snippets[0].source == "stub"
