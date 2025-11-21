from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latexify.core.data_pathway_logger import init_logger  # noqa: E402


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_data_pathway_logger_creates_files(tmp_path):
    run_dir = tmp_path / "build" / "run-test"
    logger = init_logger("test-run", run_dir, {"pdf": "inputs/sample.pdf"})

    logger.log_event(
        {
            "stage": "ingestion",
            "status": "started",
            "input_files": ["inputs/sample.pdf"],
            "models": ["nougat-small"],
            "notes": "Collecting OCR inputs",
        }
    )
    logger.log_event(
        {
            "stage": "ingestion",
            "status": "completed",
            "output_files": ["outputs/run/main.tex"],
            "notes": "Finished in 1.2s",
        }
    )
    logger.set_run_summary(
        {
            "status": "success",
            "input_pdf": "inputs/sample.pdf",
            "tex_path": "outputs/run/main.tex",
            "duration_sec": 1.2,
            "artifact_dir": "outputs/run",
        }
    )
    logger.write_human_markdown()

    jsonl_path = run_dir / "DATA_PATHWAY.llm.jsonl"
    markdown_path = run_dir / "DATA_PATHWAY.md"
    assert jsonl_path.exists()
    assert markdown_path.exists()

    events = read_jsonl(jsonl_path)
    assert len(events) == 2
    assert events[0]["stage"] == "ingestion"
    assert events[0]["status"] == "started"
    assert events[1]["output_files"] == ["outputs/run/main.tex"]

    markdown_text = markdown_path.read_text()
    assert "Data Pathway" in markdown_text
    assert "ingestion" in markdown_text.lower()
