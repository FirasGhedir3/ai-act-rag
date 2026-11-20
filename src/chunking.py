"""Stratégies de découpage comparées dans le projet.

A : blocs de taille fixe (baseline naïve, ignore la structure juridique)
B : un bloc par unité juridique (article / annexe / considérant)
C : même découpage que B ; la différence vient du reranker (voir rag.py)

Chaque chunk = {"id", "text", "metadata"}. Les métadonnées permettent de calculer
ensuite si le bon article a été retrouvé (retrieval hit-rate).
"""
from __future__ import annotations

import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingest import Unit

MAX_CHARS_B = 1400   # ~400 tokens : tient dans la fenêtre de 512 tokens du modèle d'embeddings
                     # (au-delà le modèle ignore la fin du texte). Article plus long -> redécoupé.


def chunk_fixed(units: list[Unit], chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    """Version A : texte concaténé puis découpé sans tenir compte des articles.

    On mémorise quand même, via les positions, quels articles chaque bloc recouvre :
    uniquement pour l'évaluation (le texte du bloc, lui, ne porte aucune étiquette).
    """
    spans, parts, pos = [], [], 0
    for u in units:
        body = f"{u.text}\n\n"
        spans.append((pos, pos + len(body), u.label))
        parts.append(body)
        pos += len(body)
    full = "".join(parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True
    )
    docs = splitter.create_documents([full])
    chunks = []
    for i, d in enumerate(docs):
        start = d.metadata["start_index"]
        end = start + len(d.page_content)
        covered = [lab for (s, e, lab) in spans if s < end and e > start]
        chunks.append({
            "id": f"A-{i}",
            "text": d.page_content,
            "metadata": {"variant": "A", "labels": " | ".join(covered), "label": ""},
        })
    return chunks


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Regroupe des morceaux consécutifs sans dépasser `limit` caractères."""
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 1 > limit:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def _split_paragraphs(text: str) -> list[str]:
    """Découpe un long article en respectant sa structure, du plus gros au plus fin :
    paragraphes numérotés (« 1. », « 1) »), puis points lettrés (« a) »), puis lignes."""
    parts = _pack(re.split(r"\n(?=\d+[.)]\s)", text), MAX_CHARS_B)
    out = []
    for part in parts:
        if len(part) <= MAX_CHARS_B:
            out.append(part)
            continue
        sub = _pack(re.split(r"\n(?=[a-z]\)\s)", part), MAX_CHARS_B)
        for piece in sub:
            out.extend(_pack(piece.split("\n"), MAX_CHARS_B) if len(piece) > MAX_CHARS_B else [piece])
    return out


def chunk_by_unit(units: list[Unit]) -> list[dict]:
    """Versions B et C : un chunk par article / annexe / considérant.

    Le libellé (« Article 5 — Pratiques interdites ») est répété en tête de chaque
    chunk : le LLM peut ainsi citer précisément, et l'embedding capte l'intitulé.
    """
    chunks = []
    for u in units:
        header = u.label + (f" — {u.title}" if u.title else "")
        pieces = _split_paragraphs(u.text) if len(u.text) > MAX_CHARS_B else [u.text]
        for j, piece in enumerate(pieces):
            suffix = f" (partie {j + 1}/{len(pieces)})" if len(pieces) > 1 else ""
            chunks.append({
                "id": f"B-{u.kind}-{u.number}-{j}",
                "text": f"{header}{suffix}\n{piece}",
                "metadata": {
                    "variant": "B",
                    "kind": u.kind,
                    "number": u.number,
                    "title": u.title,
                    "chapter": u.chapter,
                    "label": u.label,
                    "labels": u.label,
                },
            })
    return chunks


def include_recitals() -> bool:
    """Les considérants (exposé des motifs) ne sont pas du droit contraignant et répètent
    souvent les articles : par défaut on ne les indexe pas, pour que le chatbot cite des
    articles. INCLUDE_RECITALS=1 les ajoute (expérience de comparaison)."""
    return os.getenv("INCLUDE_RECITALS", "0") == "1"


def build_chunks(variant: str, units: list[Unit]) -> list[dict]:
    if not include_recitals():
        units = [u for u in units if u.kind != "recital"]
    if variant == "A":
        return chunk_fixed(units)
    if variant in ("B", "C"):
        return chunk_by_unit(units)
    raise ValueError(f"Variante inconnue : {variant}")
