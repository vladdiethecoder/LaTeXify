from release.models import kimi_k2_adapter as kimi


def test_discover_kimi_model_path(tmp_path, monkeypatch):
    root = tmp_path / "models"
    model_dir = root / kimi.DEFAULT_SUBDIR
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / kimi.DEFAULT_FILENAME
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(kimi, "MODELS_ROOT", root)
    monkeypatch.delenv("LATEXIFY_KIMI_K2_MODEL", raising=False)
    monkeypatch.delenv("LATEXIFY_KIMI_K2_MODEL_DIR", raising=False)
    discovered = kimi.discover_kimi_model_path()
    assert discovered == checkpoint


def test_discover_kimi_model_variant(tmp_path, monkeypatch):
    variant = "Q3_K_M"
    root = tmp_path / "models"
    model_dir = root / kimi.DEFAULT_SUBDIR
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / f"Kimi-K2-Instruct-0905-{variant}-00001-of-00002.gguf"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(kimi, "MODELS_ROOT", root)
    monkeypatch.setattr(kimi, "DEFAULT_FILENAME", f"Kimi-K2-Instruct-0905-{variant}.gguf")
    monkeypatch.delenv("LATEXIFY_KIMI_K2_MODEL", raising=False)
    monkeypatch.setenv("LATEXIFY_KIMI_K2_VARIANT", variant)
    monkeypatch.setenv("LATEXIFY_KIMI_K2_ALLOW_PATTERN", f"**/Kimi-K2-Instruct-0905-{variant}-*.gguf")
    discovered = kimi.discover_kimi_model_path()
    assert discovered == checkpoint


def test_kimi_adapter_generate_with_stub(monkeypatch, tmp_path):
    checkpoint = tmp_path / "kimi.gguf"
    checkpoint.write_text("stub", encoding="utf-8")
    calls = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def tokenize(self, data, add_bos):  # noqa: D401 - stub
            return list(range(len(data)))

        def detokenize(self, tokens):  # noqa: D401 - stub
            return b"x" * len(tokens)

        def __call__(self, **kwargs):
            calls["call"] = kwargs
            return {"choices": [{"text": "score: 0.88\nreason: tidy layout"}]}

    class FakeGrammar:
        @staticmethod
        def from_string(source: str):
            calls["grammar"] = len(source)
            return {"compiled": len(source)}

    monkeypatch.setattr(kimi, "Llama", FakeLlama)
    monkeypatch.setattr(kimi, "LlamaGrammar", FakeGrammar)
    config = kimi.GGUFModelConfig(model_path=checkpoint, default_grammar=kimi.LATEX_VALIDATION_GRAMMAR)
    adapter = kimi.KimiK2InstructAdapter(config)
    text = adapter.generate("Rate the snippet", grammar=kimi.LATEX_VALIDATION_GRAMMAR)
    assert text.startswith("score:")
    assert "prompt" in calls.get("call", {})
    assert calls.get("grammar")
    assert isinstance(adapter.warmup(), str)
