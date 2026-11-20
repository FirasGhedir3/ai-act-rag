"""Pipeline RAG : indexation Chroma, recherche (+ reranker optionnel), génération avec citations.

Variables d'environnement :
  LLM_PROVIDER = anthropic (défaut) | ollama | groq
  LLM_MODEL    = claude-haiku-5-5 (défaut anthropic) | qwen2.5:7b (défaut ollama)
  ANTHROPIC_API_KEY pour le fournisseur anthropic
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import chromadb

from .chunking import build_chunks, include_recitals
from .ingest import load_units

UNITS_PATH = Path("data/processed/units.jsonl")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-small"           # multilingue, tourne sur CPU
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # cross-encoder multilingue

SYSTEM_PROMPT = """Tu es un assistant juridique spécialisé dans le règlement européen sur l'IA \
(AI Act, règlement (UE) 2024/1689).
Règles strictes :
1. Réponds UNIQUEMENT à partir des extraits fournis dans le contexte.
2. Cite la source de chaque affirmation entre crochets, avec uniquement le libellé de l'article, \
de l'annexe ou du considérant, sans titre ni mention de « partie » : [Article 5], [Annexe III], \
[Considérant 12]. Utilise des crochets droits [ ], un seul libellé par paire.
3. Si les extraits ne permettent pas de répondre, dis-le clairement : « Les extraits fournis ne \
permettent pas de répondre. » N'invente jamais d'article ni de contenu.
4. Réponds en français, de façon concise et précise."""


# ---------------------------------------------------------------------------
# Modèles (chargés une seule fois)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    # Convention e5 : préfixe « passage: » côté documents, « query: » côté questions.
    return _embedder().encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                              batch_size=32, show_progress_bar=True).tolist()


def embed_query(q: str) -> list[float]:
    return _embedder().encode(f"query: {q}", normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Indexation
# ---------------------------------------------------------------------------
def _collection(variant: str):
    # C réutilise l'index de B : seul le reranker diffère.
    suffix = "_rec" if include_recitals() else ""   # corpus différent -> index différent
    name = f"aiact_{'B' if variant == 'C' else variant}{suffix}"
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client, name


def build_index(variant: str) -> int:
    client, name = _collection(variant)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    chunks = build_chunks(variant, load_units(UNITS_PATH))
    for i in range(0, len(chunks), 256):
        batch = chunks[i:i + 256]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embed_passages([c["text"] for c in batch]),
            metadatas=[c["metadata"] for c in batch],
        )
    return len(chunks)


# ---------------------------------------------------------------------------
# Recherche
# ---------------------------------------------------------------------------
def retrieve(question: str, variant: str = "B", k: int = 5, fetch_k: int = 20) -> list[dict]:
    client, name = _collection(variant)
    col = client.get_collection(name)
    n = fetch_k if variant == "C" else k
    res = col.query(query_embeddings=[embed_query(question)], n_results=n)
    hits = [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    if variant == "C":  # reranking : on sur-échantillonne puis on garde les k meilleurs
        scores = _reranker().predict([(question, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:k]
    return hits


# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------
def _llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("LLM_MODEL", "qwen2.5:7b"), temperature=0)
    if provider == "groq":  # offre gratuite, sans carte bancaire (clé dans GROQ_API_KEY)
        from langchain_groq import ChatGroq

        model = os.getenv("LLM_MODEL")
        if not model:
            raise SystemExit("Définis LLM_MODEL (nom d'un modèle Groq disponible sur ton compte).")
        return ChatGroq(model=model, temperature=0, max_retries=3)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=os.getenv("LLM_MODEL", "claude-haiku-5-5"), temperature=0)


def format_context(hits: list[dict]) -> str:
    # Pour la variante A les chunks n'ont pas de libellé : c'est volontaire (baseline).
    return "\n\n---\n\n".join(h["text"] for h in hits)


def answer(question: str, variant: str = "B", k: int = 5) -> dict:
    hits = retrieve(question, variant, k)
    msg = _llm().invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Contexte :\n{format_context(hits)}\n\nQuestion : {question}"),
    ])
    return {"question": question, "variant": variant, "answer": msg.content, "contexts": hits}


if __name__ == "__main__":
    import sys

    v = sys.argv[1] if len(sys.argv) > 1 else "B"
    print(f"Indexation variante {v}…")
    print(build_index(v), "chunks")
