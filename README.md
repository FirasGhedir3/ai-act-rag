# Assistant IA réglementaire (RAG) sur l'AI Act — avec évaluation de fiabilité

J'ai construit un chatbot qui répond aux questions sur le règlement (UE) 2024/1689 (AI Act) **en citant l'article source**,
et dont la fiabilité est **mesurée** sur 40 questions annotées.
J'y compare trois stratégies de recherche : **A** blocs de taille fixe · **B** un chunk par article · **C** B + reranker.

> Statut : projet en cours. Évaluation sans LLM terminée ; évaluation RAGAS en cours (quota gratuit limité) ;
> interface Streamlit et démo à venir. Les 40 questions de test doivent encore être relues une à une dans le texte officiel.

## Résultats

Jeu de test : 40 questions (36 avec réponse dans le règlement, 4 hors périmètre pour tester le refus).
Intervalles de confiance à 95 % par bootstrap (10 000 rééchantillonnages) sur les 36 questions avec réponse.
LLM de génération : `openai/gpt-oss-120b` (Groq, température 0) ; embeddings `multilingual-e5-small`.

| Métrique | A — blocs fixes | B — par article | C — B + reranker |
|---|---|---|---|
| hit@5 (le bon article est dans le top 5) | 0,89 [0,78 ; 0,97] | 0,83 [0,69 ; 0,94] | 0,86 [0,75 ; 0,97] |
| MRR (rang du bon article) | 0,72 [0,59 ; 0,84] | 0,71 [0,58 ; 0,83] | **0,85** [0,72 ; 0,94] |
| Précision des citations | 0,33 [0,17 ; 0,51] | 0,81 [0,68 ; 0,92] | 0,82 [0,71 ; 0,93] |
| Rappel des citations | 0,24 [0,11 ; 0,38] | 0,74 [0,60 ; 0,86] | 0,79 [0,67 ; 0,90] |
| Citations d'un article absent du contexte | 20 % | 0 % | 0 % |
| Refus à tort (question répondable) | 25 % | 8 % | 8 % |
| Refus à raison (hors périmètre, n = 4) | 4/4 | 4/4 | 4/4 |

**Ce que les écarts permettent de conclure** (différences appariées par question, intervalle de confiance à 95 %) :

- **B et C citent nettement mieux que A** : précision +0,48 et +0,49, rappel +0,50 et +0,56, et plus aucune citation d'un article absent des extraits. Écarts significatifs.
- **Le reranker (C) améliore le classement** : MRR +0,14 par rapport à B, intervalle [0,05 ; 0,24]. Sur le reste (hit@5, précision, rappel, refus), la différence C − B n'est **pas démontrée** avec 36 questions.
- **hit@5 : aucune différence démontrée** entre les trois versions (intervalles très chevauchants).

**Précautions de lecture**

- A a un hit@5 plus élevé en apparence, mais ses blocs ne correspondent pas à un article : A ne peut pas citer proprement « par construction ». Sa faible précision de citation reflète en partie ce choix de conception, pas seulement une faiblesse du modèle.
- 36 questions donnent des intervalles larges : les petites différences ne sont pas interprétables.
- Les résultats dépendent des articles de référence annotés à la main, en cours de relecture.
- **RAGAS** (fidélité, rappel du contexte) : en cours. Les scores partiels disponibles ne sont pas comparables entre variantes et ne sont pas publiés ici. Les refus sont exclus du calcul de fidélité, car il n'y a alors rien à vérifier.

## Comment ça marche

```
Question ─► embedding (query:) ─► Chroma (top 5)  ─► LLM (réponse + [Article N]) ─► réponse citée
                                  └ variante C : top 20 ─► cross-encoder (reranker) ─► top 5
```

1. **Ingestion** (`src/ingest.py`) : extraction du PDF officiel, nettoyage (en-têtes, notes de bas de page), découpage en 180 considérants, 113 articles et 13 annexes. Le script vérifie ces comptes.
2. **Découpage** (`src/chunking.py`) : A = blocs de 1000 caractères (chevauchement 150) ; B/C = un chunk par article, préfixé par « Article N — titre », avec découpe hiérarchique (alinéas numérotés, puis lettrés) des articles trop longs.
3. **Recherche** (`src/rag.py`) : embeddings `intfloat/multilingual-e5-small`, base vectorielle Chroma (cosinus). Variante C : reranking par `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
4. **Génération** : le LLM ne doit répondre qu'à partir des extraits, citer chaque affirmation sous la forme `[Article 5]`, et dire « Les extraits fournis ne permettent pas de répondre. » s'il ne sait pas.

## Évaluation

Les métriques sans LLM (`eval/metrics.py`) sont déterministes et reproductibles :

- **hit@5 / rappel d'article / MRR** : les articles attendus sont-ils récupérés, et à quel rang ?
- **précision et rappel des citations** : les articles cités sont-ils les bons ?
- **citation fantôme** : article cité qui n'apparaît dans aucun extrait fourni (hallucination probable) ;
- **refus à tort / à raison** : détection de la phrase de refus.

Les écarts entre variantes sont testés par bootstrap apparié (`eval/stats.py`).
RAGAS (`eval/ragas_eval.py`) ajoute fidélité et rappel du contexte avec un LLM juge ; il est plus bruité et dépend du quota de l'API.

## Choix de conception et limites

- **Considérants exclus de l'index par défaut** : au premier test, « Considérant 1 » était renvoyé à la place de « Article 1 ». Réactivables avec `INCLUDE_RECITALS=1`.
- **Taille maximale des chunks (1 400 caractères)** : fixée après avoir constaté que des articles longs (par ex. l'article 2) étaient tronqués par la fenêtre de 512 tokens du modèle d'embeddings. J'ai choisi ce réglage en regardant quelques questions de test ; il peut donc être légèrement favorable à ce jeu.
- **Écarts de vocabulaire** : certaines questions en langage courant (par ex. « chatbot ») retrouvent mal l'article juridique correspondant (article 50). Piste d'amélioration : recherche hybride BM25 + embeddings, ou reformulation de la question.
- **Jeu de test petit (40 questions)**, rédigé et annoté par moi ; relecture dans le texte officiel en cours.
- Le projet utilise uniquement des outils gratuits (Groq gratuit, modèles locaux), ce qui limite le volume d'évaluation RAGAS.
- Le système fournit une aide à la recherche dans le texte, **pas un avis juridique**.

## Utilisation

Python 3.12 recommandé.

```bash
pip install -r requirements.txt
# 1. Déposer le texte officiel (EUR-Lex, version FR) dans data/raw/ (PDF ou HTML)
python -m src.ingest data/raw/ai_act_fr.pdf     # -> data/processed/units.jsonl (+ contrôle 180/113/13)
# 2. Indexer (A et B ; C réutilise l'index de B)
python -m src.rag A && python -m src.rag B
# 3. Choisir le LLM (variables d'environnement, ne jamais mettre de clé dans le code)
#    LLM_PROVIDER=groq  LLM_MODEL=openai/gpt-oss-120b  GROQ_API_KEY=...
#    ou LLM_PROVIDER=ollama pour un modèle local
python -c "from src.rag import answer; print(answer('Quelles pratiques d IA sont interdites ?', 'C')['answer'])"
pytest -q tests
```

Évaluation :

```bash
python -m eval.run_eval --variants A B C      # pose les 40 questions (reprenable)
python -m eval.metrics                        # métriques sans LLM
python -m eval.stats                          # intervalles de confiance et différences
python -m eval.ragas_eval --variants C B A    # RAGAS (reprenable ; s'arrête quand le quota est atteint)
```

Résultats : `eval/results.sqlite` et `eval/out/*.csv` (importables dans Power BI).

## Reste à faire

- [x] Extraction validée sur le texte officiel (180 considérants / 113 articles / 13 annexes)
- [x] Trois stratégies (A/B/C), 40 questions, métriques et intervalles de confiance
- [ ] Relecture des 40 questions dans le texte officiel
- [ ] RAGAS complet pour les trois variantes avec un même LLM juge
- [ ] Interface Streamlit avec articles cités cliquables
- [ ] Tableau de bord Power BI
- [ ] Démo et déploiement

## Outils

Python 3.12, pypdf, Chroma, sentence-transformers, LangChain, Groq, RAGAS, Streamlit.

Source du texte : https://eur-lex.europa.eu/eli/reg/2024/1689/oj (version FR).
Les tests unitaires utilisent un texte synthétique qui imite la mise en forme, pas le règlement.
