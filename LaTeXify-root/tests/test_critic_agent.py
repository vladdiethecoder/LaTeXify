from __future__ import annotations

from types import SimpleNamespace

import pytest

from latexify.pipeline.critic_agent import CriticAgent
from latexify.pipeline.specialist_router import SpecialistDecision


def _decision() -> SpecialistDecision:
    prompt = SimpleNamespace(version="test")
    return SpecialistDecision(
        name="text",
        handler=lambda bundle: ("", []),
        reason="test",
        prompt=prompt,
        metadata={},
    )


def test_critic_accepts_clean_snippet():
    critic = CriticAgent(plan={"critic": {"max_attempts": 3}}, compiler=lambda snippet: (True, ""))
    result = critic.review(
        "Clean body without placeholders",
        bundle={},
        decision=_decision(),
        attempt=1,
        feedback_history=[],
    )
    assert result.accepted
    assert result.feedback == ""
    assert critic.max_attempts({"critic": {"max_attempts": 5}}) == 5


def test_critic_rejects_on_compile_failure():
    def failing_compiler(_: str):
        return False, "Missing $ sign"

    critic = CriticAgent(compiler=failing_compiler)
    result = critic.review(
        "Some math",
        bundle={"critic": {"max_attempts": 2}},
        decision=_decision(),
        attempt=1,
        feedback_history=[],
    )
    assert not result.accepted
    assert "Compilation failed" in result.feedback
    assert "1 attempt" in result.feedback


def test_critic_flags_placeholders():
    critic = CriticAgent(compiler=lambda snippet: (True, ""))
    result = critic.review(
        "% TODO fix later",
        bundle={},
        decision=_decision(),
        attempt=1,
        feedback_history=[],
    )
    assert not result.accepted
    assert "placeholder" in result.feedback.lower()


def test_compile_failure_then_fix():
    calls = {"count": 0}

    def flaky_compiler(snippet: str):
        calls["count"] += 1
        if "fixed" in snippet:
            return True, ""
        return False, "error"

    critic = CriticAgent(compiler=flaky_compiler)
    first = critic.review("broken", bundle={}, decision=_decision(), attempt=1, feedback_history=[])
    assert not first.accepted
    second = critic.review("now fixed", bundle={}, decision=_decision(), attempt=2, feedback_history=[first.feedback])
    assert second.accepted
    assert calls["count"] == 2


def test_no_compiler_available(monkeypatch):
    monkeypatch.setattr("latexify.pipeline.critic_agent.LATEXMK_BIN", None)
    critic = CriticAgent()
    result = critic.review(
        "Valid snippet",
        bundle={},
        decision=_decision(),
        attempt=1,
        feedback_history=[],
    )
    assert result.accepted
