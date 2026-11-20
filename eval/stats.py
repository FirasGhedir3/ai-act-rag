"""Étape 4 : incertitude statistique (intervalles de confiance bootstrap + comparaisons appariées).

Avec 40 questions, un écart de quelques points peut être dû au hasard. On rééchantillonne
les questions (avec remise, 10 000 fois, le même tirage pour toutes les variantes) pour
estimer un intervalle de confiance à 95 %. Une différence n'est dite « réelle » que si son
intervalle exclut 0.

Usage (après python -m eval.metrics) : python -m eval.stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("eval/out")
METRICS = ["hit_at_k", "mrr", "citation_precision", "citation_recall",
           "false_abstention", "ghost_citation", "correct_abstention"]
PAIRS = [("B", "A"), ("C", "B"), ("C", "A")]   # (variante, référence) : différence = premier - second
N_BOOT = 10_000


def main() -> None:
    df = pd.read_csv(OUT / "metrics_by_question.csv")
    rng = np.random.default_rng(0)
    ci_rows, diff_rows = [], []
    for m in METRICS:
        wide = df.pivot(index="qid", columns="variant", values=m).dropna(how="all")
        if wide.empty:
            continue
        x = wide.to_numpy(dtype=float)                       # questions x variantes
        idx = rng.integers(0, len(wide), size=(N_BOOT, len(wide)))
        with np.errstate(all="ignore"):
            boots = np.nanmean(x[idx], axis=1)               # N_BOOT x variantes
        cols = list(wide.columns)
        for j, v in enumerate(cols):
            lo, hi = np.nanpercentile(boots[:, j], [2.5, 97.5])
            ci_rows.append({"metric": m, "variant": v, "n": int(wide[v].notna().sum()),
                            "mean": round(float(np.nanmean(x[:, j])), 3), "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)})
        for a, b in PAIRS:
            if a in cols and b in cols:
                d = boots[:, cols.index(a)] - boots[:, cols.index(b)]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                diff_rows.append({"metric": m, "comparison": f"{a} - {b}",
                                  "diff": round(float(np.nanmean(x[:, cols.index(a)]) - np.nanmean(x[:, cols.index(b)])), 3),
                                  "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                                  "reel": "oui" if (lo > 0 or hi < 0) else "non (incertain)"})
    ci, diffs = pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)
    ci.to_csv(OUT / "stats_ci.csv", index=False)
    diffs.to_csv(OUT / "stats_diffs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("=== Valeurs avec intervalle de confiance à 95 % ===\n", ci.to_string(index=False))
    print("\n=== Différences entre variantes ===\n", diffs.to_string(index=False))


if __name__ == "__main__":
    main()
