from eval.metrics import cited_labels, is_abstention, score_row

Q_OK = {"answerable": True, "expected_articles": ["Article 18"]}
Q_NO = {"answerable": False, "expected_articles": []}


def test_cited_labels_variants():
    a = "Dix ans [Article 18]. Voir aussi [Annexe III] et [Article 5, paragraphe 1] et [Considérant 12]."
    assert cited_labels(a) == ["Article 18", "Annexe III", "Article 5", "Considérant 12"]
    assert cited_labels("Pas de source ici.") == []


def test_abstention_detection():
    assert is_abstention("Les extraits fournis ne permettent pas de répondre.")
    assert is_abstention("Cette information ne figure pas dans le texte.")
    assert not is_abstention("Le délai est de dix ans [Article 18].")


def test_score_row_answerable():
    ctx = [["Article 5"], ["Article 18"]]
    s = score_row(Q_OK, "10 ans [Article 18]", ctx)
    assert s["hit_at_k"] == 1 and s["mrr"] == 0.5
    assert s["citation_precision"] == 1.0 and s["citation_recall"] == 1.0
    assert s["ghost_citation"] == 0 and s["false_abstention"] == 0


def test_ghost_citation_and_miss():
    s = score_row(Q_OK, "Voir [Article 99]", [["Article 5"]])
    assert s["hit_at_k"] == 0 and s["mrr"] == 0.0
    assert s["ghost_citation"] == 1 and s["citation_recall"] == 0.0


def test_unanswerable():
    assert score_row(Q_NO, "Les extraits ne permettent pas de répondre.", [["Article 5"]])["correct_abstention"] == 1
    assert score_row(Q_NO, "L'amende est de 5 M€ [Article 99].", [["Article 5"]])["correct_abstention"] == 0


def test_fullwidth_brackets():
    assert cited_labels("Exclus du champ【Considérant 24】 et [Article 2].") == ["Considérant 24", "Article 2"]


def test_ghost_ignores_cross_references():
    # « Article 12 » n'est pas retrouvé mais le texte d'un passage y renvoie : pas une invention
    ctx = ["Les journaux visés à l'article 12 sont conservés."]
    s = score_row(Q_OK, "Voir [Article 12]", [["Article 26"]], ctx)
    assert s["unretrieved_citation"] == 1 and s["ghost_citation"] == 0
    s = score_row(Q_OK, "Voir [Article 5]", [["Article 26"]], ctx)
    assert s["ghost_citation"] == 1
