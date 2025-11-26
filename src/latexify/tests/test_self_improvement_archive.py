import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from latexify.self_improvement.archive import Archive
from latexify.self_improvement.models import AgentVersion, AgentMetrics
from latexify.self_improvement.patching import PatchOperation, PatchProposal, apply_patch_proposal
from latexify.self_improvement.thought_log import ThoughtLog, ThoughtEntry


def test_archive_selection_biases_toward_higher_score():
    archive = Archive(
        versions=[
            AgentVersion(version_id="low", parent_id=None, strategy="TARGETED", metrics=AgentMetrics(score=0.1)),
            AgentVersion(version_id="high", parent_id=None, strategy="TARGETED", metrics=AgentMetrics(score=0.9)),
        ]
    )
    picks = [archive.select_parent(temperature=0.5).version_id for _ in range(100)]
    assert picks.count("high") > picks.count("low")


def test_patch_operations_apply_and_log(tmp_path: Path):
    target = tmp_path / "file.txt"
    target.write_text("hello world", encoding="utf-8")

    proposal = PatchProposal(
        candidate_id="c1",
        strategy="TARGETED",
        rationale="replace greeting",
        operations=[PatchOperation(file_path=Path("file.txt"), search="hello", replace="hi")],
    )

    result = apply_patch_proposal(proposal, tmp_path)
    assert result.success
    assert target.read_text(encoding="utf-8") == "hi world"

    log = ThoughtLog(tmp_path / "log.md")
    entry = ThoughtEntry(
        iteration=1,
        agent_id="c1",
        strategy=proposal.strategy,
        summary="test",
        details={"accepted": True},
    )
    log.append(entry)
    assert (tmp_path / "log.md").exists()
