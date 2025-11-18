PYTHON ?= python3
BACKEND_PORT ?= 8000

.PHONY: env backend ui dev test training-route prepare-training-data finetune-latex-lm

env:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e .

backend:
	uvicorn backend.app.main:app --reload --port $(BACKEND_PORT)

ui:
	streamlit run apps/ui/streamlit_app.py

dev:
	bash -c "uvicorn backend.app.main:app --reload --port $(BACKEND_PORT) & PID=$$!; streamlit run apps/ui/streamlit_app.py; kill $$PID"

test:
	pytest -q

training-route:
	$(PYTHON) scripts/build_training_data_route.py

prepare-training-data:
	$(PYTHON) scripts/download_all_training_data.py

finetune-latex-lm:
	$(PYTHON) scripts/finetune_latex_lm.py --dataset-slug mathbridge --model-name Qwen/Qwen2.5-0.5B-Instruct --max-records 128 --dry-run
