from latexify.pipeline.aggregator import choose_engine, latexmk_flags_for_engine

def test_choose_engine_lix_defaults_to_xelatex():
    assert choose_engine("textbook") == "xelatex"

def test_flags_for_xelatex():
    flags = latexmk_flags_for_engine("xelatex")
    assert "-pdfxe" in flags and "-g" in flags
