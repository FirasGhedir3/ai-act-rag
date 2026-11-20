"""Étape 1 de l'évaluation : poser les questions au chatbot, variante par variante.

Chaque réponse (+ les passages retrouvés) est stockée dans SQLite (eval/results.sqlite).
Le script est REPRENABLE : si tu l'arrêtes (limite de l'API, veille…), relance la même
commande, il saute les questions déjà traitées.

Usage (depuis la racine du projet, environnement activé) :
    python -m eval.run_eval --variants A B C
    python -m eval.run_eval --variants B --limit 3      # test rapide sur 3 questions
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

QUESTIONS = Path("eval/questions.jsonl")
DB = Path("eval/results.sqlite")


def load_questions(limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def init_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            variant TEXT, qid TEXT, question TEXT, answer TEXT,
            contexts_json TEXT, retrieved_labels_json TEXT,
            latency_s REAL, created_at TEXT,
            PRIMARY KEY (variant, qid))"""
    )
    return con


def labels_of(hit: dict) -> list[str]:
    """Articles couverts par un passage (A : plusieurs possibles ; B/C : un seul)."""
    return [l for l in hit["metadata"].get("labels", "").split(" | ") if l]


def call_with_retry(fn, *args, attempts: int = 6):
    for i in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception as e:  # limite de débit (429), réseau…
            wait = 20 * i
            print(f"   ⚠ erreur ({type(e).__name__}: {str(e)[:90]}) – nouvel essai dans {wait}s")
            time.sleep(wait)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pause", type=float, default=3.0, help="secondes entre deux questions")
    args = ap.parse_args()

    from src.rag import answer  # import tardif : charge les modèles d'embeddings

    questions = load_questions(args.limit)
    con = init_db()
    for v in args.variants:
        done = {r[0] for r in con.execute("SELECT qid FROM runs WHERE variant=?", (v,))}
        todo = [q for q in questions if q["id"] not in done]
        print(f"=== Variante {v} : {len(todo)} question(s) à traiter ({len(done)} déjà faites)")
        for q in todo:
            t0 = time.time()
            res = call_with_retry(answer, q["question"], v, args.k)
            if res is None:
                print(f"   ✗ {q['id']} abandonnée, relance le script plus tard")
                continue
            con.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,datetime('now'))",
                (v, q["id"], q["question"], res["answer"],
                 json.dumps([h["text"] for h in res["contexts"]], ensure_ascii=False),
                 json.dumps([labels_of(h) for h in res["contexts"]], ensure_ascii=False),
                 round(time.time() - t0, 2)),
            )
            con.commit()
            print(f"   ✓ {q['id']} ({time.time() - t0:.1f}s)")
            time.sleep(args.pause)
    con.close()


if __name__ == "__main__":
    main()
