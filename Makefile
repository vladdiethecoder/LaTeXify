PYTHON ?= python3
BACKEND_PORT ?= 8000

.PHONY: env backend ui dev test

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
