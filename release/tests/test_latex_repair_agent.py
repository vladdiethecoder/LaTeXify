from release.pipeline.latex_repair_agent import KimiK2LatexRepair


def test_preflight_balances_and_packages(tmp_path):
    tex_path = tmp_path / "doc.tex"
    tex_path.write_text("\\documentclass{article}\n\\begin{document}\n\\begin{itemize}\n\\item A\n", encoding="utf-8")
    agent = KimiK2LatexRepair()
    report = agent.preflight_check(tex_path)
    text = tex_path.read_text(encoding="utf-8")
    assert "\\end{itemize}" in text
    assert isinstance(report["packages"], list)


def test_diagnose_and_package_suggestion():
    agent = KimiK2LatexRepair()
    log = "! LaTeX Error: Environment align undefined.\nUndefined control sequence."
    diagnostics = agent.diagnose_log(log)
    assert any(rec.code == "environment" for rec in diagnostics)
    packages = agent.suggest_packages("Use \\includegraphics{foo}")
    assert "graphicx" in packages


def test_repair_agent_honors_env_overrides(monkeypatch):
    monkeypatch.setenv("LATEXIFY_KIMI_K2_TEMPERATURE", "0.33")
    monkeypatch.setenv("LATEXIFY_KIMI_K2_MAX_TOKENS", "512")
    agent = KimiK2LatexRepair()
    assert agent.temperature == 0.33
    assert agent.max_tokens == 512
