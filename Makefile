UV ?= uv
PYTHON ?= python3
BACKEND_PORT ?= 8000

.PHONY: env sync lint format typecheck test backend ui dev pre-commit-install training-route prepare-training-data finetune-latex-lm

env:
	$(UV) venv .venv --python $(PYTHON)

sync:
	$(UV) sync

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy .

test:
	$(UV) run pytest

backend:
	$(UV) run uvicorn backend.app.main:app --reload --port $(BACKEND_PORT)

ui:
	$(UV) run streamlit run apps/ui/streamlit_app.py

dev:
	bash -c "$(UV) run uvicorn backend.app.main:app --reload --port $(BACKEND_PORT) & PID=$$!; $(UV) run streamlit run apps/ui/streamlit_app.py; kill $$PID"

pre-commit-install:
	$(UV) run pre-commit install

training-route:
	$(UV) run python scripts/build_training_data_route.py

prepare-training-data:
	$(UV) run python scripts/download_all_training_data.py

finetune-latex-lm:
	$(UV) run python scripts/finetune_latex_lm.py --dataset-slug mathbridge --model-name Qwen/Qwen2.5-0.5B-Instruct --max-records 128 --dry-run
