"""Tests sur un texte SYNTHÉTIQUE qui imite la mise en forme d'EUR-Lex (pas le vrai règlement)."""
from src.ingest import parse_units
from src.chunking import chunk_by_unit, chunk_fixed

SAMPLE = """
considérant ce qui suit:
(1) Premier considérant fictif sur le marché intérieur.
(2) Deuxième considérant fictif, qui renvoie à l'article 5 dans le fil du texte.
CHAPITRE I
DISPOSITIONS GÉNÉRALES
Article 1
Objet
1. Le présent règlement a pour objet fictif A.
2. Il vise aussi l'objectif fictif B.
Article 2
Champ d'application
1. Le présent règlement s'applique, comme le précise l'article 1, aux fournisseurs fictifs.
CHAPITRE II
PRATIQUES INTERDITES
Article 5
Pratiques d'IA interdites
1. Les pratiques suivantes sont interdites :
(a) pratique fictive un ;
(b) pratique fictive deux.
ANNEXE I
Liste fictive
1. Élément un.
ANNEXE III
Autre liste fictive
1. Élément trois.
"""

def test_parse_counts_and_titles():
    units = parse_units(SAMPLE)
    by = {(u.kind, u.number): u for u in units}
    assert [u.number for u in units if u.kind == "recital"] == ["1", "2"]
    assert set(k for k in by if k[0] == "article") == {("article", "1"), ("article", "2"), ("article", "5")}
    assert set(k for k in by if k[0] == "annex") == {("annex", "I"), ("annex", "III")}
    assert by[("article", "5")].title == "Pratiques d'IA interdites"
    assert by[("article", "5")].chapter == "CHAPITRE II"
    assert by[("article", "1")].chapter == "CHAPITRE I"

def test_cross_references_do_not_split():
    units = parse_units(SAMPLE)
    art2 = next(u for u in units if u.kind == "article" and u.number == "2")
    assert "l'article 1" in art2.text          # renvoi conservé dans le texte
    assert len([u for u in units if u.kind == "article"]) == 3

def test_chunk_by_unit_keeps_label():
    chunks = chunk_by_unit(parse_units(SAMPLE))
    c = next(c for c in chunks if c["metadata"]["label"] == "Article 5")
    assert c["text"].startswith("Article 5 — Pratiques d'IA interdites")

def test_chunk_fixed_tracks_covered_articles():
    chunks = chunk_fixed(parse_units(SAMPLE), chunk_size=200, overlap=20)
    assert any("Article 5" in c["metadata"]["labels"] for c in chunks)
