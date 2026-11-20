"""Étape 3 : métriques RAGAS (un LLM sert de juge).

  faithfulness     la réponse est-elle fidèle aux passages fournis ? (mesure les hallucinations)
  context_recall   les passages retrouvés contiennent-ils l'information de la réponse de référence ?

Seules les questions AVEC réponse sont évaluées (la référence des questions hors
périmètre n'est pas une vraie réponse). Le script est REPRENABLE (scores gardés dans SQLite)
et lent par choix : un seul appel à la fois pour rester dans les limites de l'offre gratuite.

Usage : python -m eval.ragas_eval --variants A B C [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

QUESTIONS = Path("eval/questions.jsonl")
DB = Path("eval/results.sqlite")
OUT = Path("eval/out")
BATCH = 5  # on sauvegarde après chaque lot de 5 questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--limit", type=int, default=None, help="nb de questions par variante (test)")
    args = ap.parse_args()

    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall  # anciennes classes : avertissement possible
    from ragas.run_config import RunConfig

    from src.rag import _llm

    judge = LangchainLLMWrapper(_llm())
    run_config = RunConfig(max_workers=1, timeout=240, max_retries=10)

    questions = {q["id"]: q for q in map(json.loads, QUESTIONS.read_text(encoding="utf-8").splitlines())}
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS ragas_scores (
        variant TEXT, qid TEXT, faithfulness REAL, context_recall REAL, PRIMARY KEY (variant, qid))""")

    for v in args.variants:
        done = {r[0] for r in con.execute(
            "SELECT qid FROM ragas_scores WHERE variant=? AND faithfulness IS NOT NULL "
            "AND context_recall IS NOT NULL", (v,))}   # les scores vides seront retentés
        runs = pd.read_sql("SELECT * FROM runs WHERE variant=?", con, params=(v,))
        todo = [r for r in runs.itertuples() if questions[r.qid]["answerable"] and r.qid not in done]
        todo = todo[: args.limit] if args.limit else todo
        print(f"=== Variante {v} : {len(todo)} question(s) à évaluer")
        for i in range(0, len(todo), BATCH):
            lot = todo[i:i + BATCH]
            ds = EvaluationDataset(samples=[
                SingleTurnSample(
                    user_input=r.question, response=r.answer,
                    retrieved_contexts=json.loads(r.contexts_json),
                    reference=questions[r.qid]["reference_answer"],
                ) for r in lot])
            res = evaluate(dataset=ds, metrics=[Faithfulness(), LLMContextRecall()],
                           llm=judge, run_config=run_config, show_progress=False).to_pandas()
            vals = [(_num(s.get("faithfulness")), _num(s.get("context_recall"))) for _, s in res.iterrows()]
            for r, (f, c) in zip(lot, vals):
                old_f, old_c = con.execute(
                    "SELECT faithfulness, context_recall FROM ragas_scores WHERE variant=? AND qid=?",
                    (v, r.qid)).fetchone() or (None, None)
                # on ne remplace jamais un score valide par un score vide
                con.execute("INSERT OR REPLACE INTO ragas_scores VALUES (?,?,?,?)",
                            (v, r.qid, f if f is not None else old_f, c if c is not None else old_c))
            con.commit()
            n_vides = sum(x is None for pair in vals for x in pair)
            print(f"   lot {i // BATCH + 1} enregistré ({n_vides} score(s) vide(s) sur {2 * len(vals)})")
            if n_vides > len(vals):  # plus de la moitié ont échoué : quota ou réseau
                print("\n⚠ Trop d'échecs : limite quotidienne de l'API probablement atteinte.\n"
                      "  Arrêt pour ne pas gaspiller le quota. Relance cette commande plus tard :\n"
                      "  les scores déjà valides sont conservés, seuls les manquants seront retentés.")
                break

    scores, summary = summarize(con)
    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / "ragas_by_question.csv", index=False)
    summary.to_csv(OUT / "ragas_summary.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== RAGAS par variante ===\n", summary)
    con.close()


def summarize(con):
    """Moyennes RAGAS par variante. Un refus (« les extraits ne permettent pas de répondre »)
    n'a rien à vérifier : RAGAS le note 0 en fidélité. On le retire de la fidélité."""
    scores = pd.read_sql("SELECT * FROM ragas_scores", con)
    try:
        ab = pd.read_sql("SELECT variant, qid, abstained FROM metrics_by_question", con)
        scores = scores.merge(ab, on=["variant", "qid"], how="left")
    except Exception:  # metrics.py pas encore lancé
        scores["abstained"] = 0
    scores["abstained"] = scores["abstained"].fillna(0)
    rows = {}
    for v, g in scores.groupby("variant"):
        ok = g[g["abstained"] == 0]
        rows[v] = {"n_questions": len(g),
                   "faithfulness_valides": int(g["faithfulness"].notna().sum()),
                   "context_recall_valides": int(g["context_recall"].notna().sum()),
                   "faithfulness_hors_refus": round(ok["faithfulness"].mean(), 3),
                   "n_hors_refus": int(ok["faithfulness"].notna().sum()),
                   "context_recall": round(g["context_recall"].mean(), 3)}
    return scores, pd.DataFrame(rows).T


def _num(x):
    try:
        x = float(x)
        return None if x != x else x  # NaN -> NULL
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
