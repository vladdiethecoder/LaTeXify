# TO-IMPROVE

- `release/models/model_adapters.py`: explore more granular VRAM budgeting (per-layer placement or flash-attn2 integration) so Florence2/InternVL no longer need to fall back to full CPU offload when a single shard tips over the limit.
- `release/pipeline/specialists.py`: the new `question` agent outputs tidy boxes, but we still lack an `answer` agent that can summarize student work or build a rubric/table automatically.
- `release/pipeline/reward_mm.py`: now that InternVL scoring is hooked up, add a second pass that compares the rendered PDF to the source PDF (SSIM/LPIPS) to catch missing paragraphs, not just aesthetic drift.
