# 0) Start dev stack with GPUs mounted and volumes to kb/, runs/
docker compose -f dev/compose/docker-compose.dev.yml up -d

# 1) Create a tiny Course KB & ingest sample files
bash dev/scripts/ingest_sample.sh --in dev/fixtures/inbox --kb kb/course/GEN_BASELINE

# 2) Build FAISS index
python dev/scripts/build_index.py --kb kb/course/GEN_BASELINE

# 3) Run the LangGraph pipeline (Planner↔ClassChooser→Compose)
python dev/scripts/run_graph.py +profiles=math_practice paths.kb=kb/course/GEN_BASELINE

# 4) Build PDF with latexmk
bash dev/scripts/build_pdf.sh outputs/GEN_BASELINE/main.tex

# 5) Vision-QA + auto-fix loop
python dev/scripts/vision_qa.py dev/runs/<timestamp>/pdf/*.pdf
python dev/scripts/patch_apply.py dev/runs/<timestamp>/defects.json
