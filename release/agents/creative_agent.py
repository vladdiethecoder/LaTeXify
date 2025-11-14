"""Creative agent that drafts LaTeX given chunk content + research snippets."""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import shorten

from .graph_state import GraphState


@dataclass
class CreativeAgent:
    """Simple heuristic creative agent; placeholder for a future LLM node."""

    max_context_chars: int = 400

    def propose(self, state: GraphState) -> GraphState:
        """Generate a draft snippet by weaving chunk text + research hints."""

        context = shorten(state.content.strip(), width=self.max_context_chars, placeholder=" …")
        research = "\n".join(state.research_snippets[-2:]) if state.research_snippets else ""
        snippet_lines = [
            "% CreativeAgent draft",
            "\\begin{aligned}",
            context.replace("\n", " "),
            "\\end{aligned}",
        ]
        if research:
            snippet_lines.append(f"% research hint:\\n% {research}")
        state.candidate_latex = "\n".join(snippet_lines)
        state.log("creative: drafted snippet")
        return state


__all__ = ["CreativeAgent"]
