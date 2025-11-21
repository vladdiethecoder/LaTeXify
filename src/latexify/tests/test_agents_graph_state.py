from latexify.agents.graph_state import ContentChunk, GraphState, AGENT_PIPELINE_STAGE_MAP


def test_graph_state_helpers_track_pipeline_stages(tmp_path):
    state = GraphState(chunk_id="c42", content="E=mc^2")
    state.mark_stage("creative", notes="draft")
    state.record_metrics(tokens=42)
    artifact_path = tmp_path / "snippet.tex"
    artifact_path.write_text("test", encoding="utf-8")
    state.attach_artifact("draft", str(artifact_path))

    assert state.stage_history == ["branch_b_vision:draft"]
    assert state.metrics == {"tokens": 42.0}
    assert state.artifacts["draft"].endswith("snippet.tex")


def test_content_chunk_dataclass():
    chunk = ContentChunk(chunk_id="p1", text="content", metadata={"page": 1})
    assert chunk.metadata["page"] == 1


def test_agent_stage_map_covers_known_agents():
    assert AGENT_PIPELINE_STAGE_MAP["creative"] == "branch_b_vision"
    assert "compile" in AGENT_PIPELINE_STAGE_MAP
