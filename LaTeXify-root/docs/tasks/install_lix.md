# Install LiX classes locally (no root needed)

Goal: make classes like `textbook.cls`, `thesis.cls`, `paper.cls`, `novel.cls`, etc. visible to TeX, so `\documentclass{textbook}` works and the aggregator won’t fall back.

## Where to put `.cls`/`.sty`
Use your **personal TEXMF tree**. Ask TeX where it is:
```bash
kpsewhich -var-value TEXMFHOME
