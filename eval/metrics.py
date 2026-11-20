"""Étape 2 : métriques sans LLM (gratuites, déterministes, reproductibles).

  hit_at_k            le bon article figure-t-il parmi les passages retrouvés ?
  article_recall      part des articles attendus retrouvés
  mrr                 1/rang du premier passage utile (1.0 = en première position)
  citation_precision  parmi les articles cités dans la réponse, part des bons
  citation_recall     part des articles attendus effectivement cités
  ghost_citation      1 si la réponse cite un article ABSENT des passages fournis ET jamais
                      mentionné dans leur texte (= source inventée / tirée de la mémoire du modèle)
  unretrieved_citation  1 si elle cite un article qui n'est pas parmi les passages retrouvés
                      (inclut les renvois : un passage qui dit « voir l'article 12 »)
  correct_abstention  (questions hors périmètre) le chatbot dit-il qu'il ne sait pas ?
  false_abstention    (questions avec réponse) refuse-t-il à tort de répondre ?

Usage : python -m eval.metrics
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

QUESTIONS = Path("eval/questions.jsonl")
DB = Path("eval/results.sqlite")
OUT = Path("eval/out")

CITE_RE = re.compile(r"[\[【]([^\]】]+)[\]】]")   # accepte [ ] et 【 】 (utilisés par certains modèles)
LABEL_RE = re.compile(r"(Article|Annexe|Considérant)s?\s+(\d+|[IVXL]+)", re.I)
ABSTAIN_RE = re.compile(
    r"ne permettent? pas de répondre|ne figure(?:nt)? pas|ne contien(?:nt|t) pas|"
    r"aucune information|impossible de répondre|pas (?:d'|de )informations?|ne mentionn(?:e|ent) pas",
    re.I,
)


def cited_labels(answer: str) -> list[str]:
    out = []
    for group in CITE_RE.findall(answer):
        for kind, num in LABEL_RE.findall(group):
            out.append(f"{kind.capitalize()} {num.upper()}")
    return out


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer.replace("’", "'")))


def mentioned_in(label: str, contexts: list[str]) -> bool:
    """Le libellé (ex. « Article 12 ») est-il écrit dans le texte des passages (renvoi) ?"""
    kind, num = label.split()
    pat = re.compile(rf"{kind[:5]}\w*\s+{re.escape(num)}\b", re.I)
    return any(pat.search(c) for c in contexts)


def score_row(q: dict, answer: str, retrieved_labels: list[list[str]], contexts: list[str] | None = None) -> dict:
    expected = set(q["expected_articles"])
    cited = set(cited_labels(answer))
    retrieved = {l for ctx in retrieved_labels for l in ctx}
    abst = is_abstention(answer)
    missing = cited - retrieved
    invented = {l for l in missing if not mentioned_in(l, contexts or [])}
    row = {"abstained": int(abst), "n_cited": len(cited),
           "unretrieved_citation": int(bool(missing)), "ghost_citation": int(bool(invented))}
    if q["answerable"]:
        rank = next((i + 1 for i, ctx in enumerate(retrieved_labels) if expected & set(ctx)), None)
        row.update(
            hit_at_k=int(bool(expected & retrieved)),
            article_recall=len(expected & retrieved) / len(expected),
            mrr=1 / rank if rank else 0.0,
            citation_precision=len(cited & expected) / len(cited) if cited else None,
            citation_recall=len(cited & expected) / len(expected),
            false_abstention=int(abst),
        )
    else:
        row.update(correct_abstention=int(abst))
    return row


def main() -> None:
    questions = {q["id"]: q for q in map(json.loads, QUESTIONS.read_text(encoding="utf-8").splitlines())}
    con = sqlite3.connect(DB)
    runs = pd.read_sql("SELECT * FROM runs", con)
    rows = []
    for r in runs.itertuples():
        q = questions[r.qid]
        s = score_row(q, r.answer, json.loads(r.retrieved_labels_json), json.loads(r.contexts_json))
        rows.append({"variant": r.variant, "qid": r.qid, "type": q["type"], "answerable": int(q["answerable"]),
                     "latency_s": r.latency_s, **s})
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("variant", "qid", "type", "answerable")]
    summary = df.groupby("variant")[metric_cols].mean(numeric_only=True).round(3)
    summary["latency_s"] = df.groupby("variant")["latency_s"].median().round(1)  # médiane : robuste aux pauses/réessais
    summary.insert(0, "n_questions", df.groupby("variant").size())
    by_type = df.groupby(["variant", "type"])[["hit_at_k", "mrr", "citation_recall", "ghost_citation"]].mean().round(3)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "metrics_by_question.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv")
    by_type.to_csv(OUT / "metrics_by_type.csv")
    df.to_sql("metrics_by_question", con, if_exists="replace", index=False)
    summary.reset_index().to_sql("metrics_summary", con, if_exists="replace", index=False)
    con.close()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== Résumé par variante ===\n", summary.T)
    print("\n=== Par type de question ===\n", by_type)
    print(f"\nFichiers écrits dans {OUT}/ (utilisables dans Power BI)")


if __name__ == "__main__":
    main()
