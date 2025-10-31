from scripts.docclass_map import normalize_docclass

def test_normalize_lix_aliases():
    for alias in ["lix", "lix_article", "lix_textbook"]:
        d = normalize_docclass(alias)
        assert d.normalized == "textbook"
        assert d.requires_lix is True

def test_passthroughs():
    for cls in ["scrartcl", "article", "textbook"]:
        d = normalize_docclass(cls)
        assert d.normalized == cls
