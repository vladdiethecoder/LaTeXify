# TODO

- [ ] `backend/app/graphs/runner.py`: replace the placeholder `GraphRunner` with an actual LangGraph or pydantic-based state machine so the SSE endpoint can be flipped from mock mode.
- [ ] `release/pipeline/ingestion.py`: add unit tests that cover the new `LayoutAnalyzer` classification (questions, tables, multi-column detection) to keep the heuristics from regressing.
- [ ] `apps/ui/streamlit_app.py`: surface `block_done` events visually (e.g., checkmarks or fading placeholders) so operators can tell when each specialist is finished streaming.
