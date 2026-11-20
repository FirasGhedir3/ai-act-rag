# Assistant IA réglementaire (RAG) sur l'AI Act — avec évaluation de fiabilité

Chatbot qui répond aux questions sur le règlement (UE) 2024/1689 en citant l'article source,
et dont la fiabilité est mesurée (RAGAS) sur un jeu de 40 questions annotées.
Trois stratégies comparées : **A** blocs fixes · **B** découpage par article · **C** B + reranker.

## Démarrage
```bash
pip install -r requirements.txt
# 1. Déposer le texte officiel (EUR-Lex, version FR) dans data/raw/ (PDF ou HTML)
python -m src.ingest data/raw/ai_act_fr.pdf     # -> data/processed/units.jsonl (+ contrôle 180/113/13)
# 2. Indexer (A et B ; C réutilise l'index de B)
python -m src.rag A && python -m src.rag B
# 3. Tester
export ANTHROPIC_API_KEY=...        # ou LLM_PROVIDER=ollama
python -c "from src.rag import answer; print(answer('Quelles pratiques d IA sont interdites ?', 'C')['answer'])"
pytest -q tests
```

## Évaluation
```bash
python -m eval.run_eval --variants A B C      # pose les 40 questions (reprenable)
python -m eval.metrics                        # métriques sans LLM : retrieval, citations, abstention
python -m eval.stats                          # intervalles de confiance et différences entre variantes
python -m eval.ragas_eval --variants A B C    # RAGAS : faithfulness, context_recall
```
Résultats : `eval/results.sqlite` et `eval/out/*.csv` (importables dans Power BI).

## Avancement
- [x] Semaine 1 — code d'extraction, découpage (A/B), indexation Chroma, génération avec citations
- [x] Semaine 1 — extraction validée sur le texte officiel (180 considérants / 113 articles / 13 annexes)
- [x] Semaine 2 — 40 questions de test (`eval/questions.jsonl`, à relire dans le texte officiel)
- [x] Semaine 3 — scripts d'évaluation écrits (`eval/`), à exécuter
- [ ] Semaines 3-6 — RAGAS, versions B/C, Streamlit, Power BI, démo

Source du texte : https://eur-lex.europa.eu/eli/reg/2024/1689/oj (version FR).
Les tests utilisent un texte synthétique qui imite la mise en forme, pas le règlement.
