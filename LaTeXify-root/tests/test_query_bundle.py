from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from latexify.kb.query_index import build_context_bundle


def _mk_dummy_index(tmp: Path, name: str, n: int = 8, d: int = 32) -> Path:
    # deterministic random vectors
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, d)).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    idx = faiss.IndexFlatIP(d)
    idx.add(X)
    idir = tmp / name
    idir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(idir / "faiss.index"))
    metas = [{"id": f"{name}_{i}", "page": i + 1, "label": name, "source_image": f"{name}.pdf", "ocr_model": None, "bbox": None}
             for i in range(n)]
    (idir / "faiss.meta.json").write_text(
        json.dumps({"dim": d, "size": n, "ids": [m["id"] for m in metas], "metas": metas, "model": "dummy"}, indent=2),
        encoding="utf-8",
    )
    return idir


def test_build_context_bundle_offline(tmp_path: Path, monkeypatch):
    a = _mk_dummy_index(tmp_path, "assignment")
    r = _mk_dummy_index(tmp_path, "rubric")
    u = _mk_dummy_index(tmp_path, "user")
    s = _mk_dummy_index(tmp_path, "assessment")

    # ensure no network embedding is attempted
    monkeypatch.setenv("DISABLE_ST_EMBEDDING", "1")

    task = {"id": "T99", "question": "How should tables be formatted with booktabs & siunitx?"}
    bundle = build_context_bundle(task, {"assignment": str(a), "rubric": str(r), "user": str(u), "assessment": str(s)}, k_user=3)
    assert isinstance(bundle, dict) and "user_answer" in bundle
    ev = Path("evidence") / "T99.json"
    assert ev.exists() and ev.stat().st_size > 0
