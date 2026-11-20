"""Extraction et structuration de l'AI Act (règlement (UE) 2024/1689).

Entrée  : un fichier officiel EUR-Lex (.pdf, .html ou .txt) placé dans data/raw/.
Sortie  : data/processed/units.jsonl, une ligne par unité juridique
          (considérant, article ou annexe) avec ses métadonnées.

Usage : python -m src.ingest data/raw/ai_act_fr.pdf
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Unit:
    kind: str          # "recital" | "article" | "annex"
    number: str        # "5", "XIII", "12" (considérant)
    title: str         # intitulé de l'article / de l'annexe ("" pour un considérant)
    chapter: str       # ex. "CHAPITRE II" ("" si inconnu)
    text: str          # texte complet de l'unité

    @property
    def label(self) -> str:
        """Étiquette de citation, ex. 'Article 5', 'Annexe III', 'Considérant 12'."""
        return {"article": "Article", "annex": "Annexe", "recital": "Considérant"}[self.kind] + f" {self.number}"


# ----------------------------------------------------------------------------
# 1. Lecture du fichier brut -> texte
# ----------------------------------------------------------------------------
def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if suffix in {".html", ".htm", ".xhtml"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Dans le HTML EUR-Lex, "1." et le texte du paragraphe sont dans deux cellules :
        # on les recolle pour garder un texte lisible.
        text = soup.get_text("\n")
        return re.sub(r"\n(\d+\.|\([a-z0-9]+\))\n", r"\n\1 ", text)
    return path.read_text(encoding="utf-8", errors="ignore")


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace(" ", " ")
    # En-têtes / pieds de page du Journal officiel répétés dans le PDF
    text = re.sub(r"^\s*(JO L,? \d+\.\d+\.\d{4}|\d+/\d+\s*)$", "", text, flags=re.M)
    text = re.sub(r"^\s*ELI:.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*FR\s*$|^\s*EN\s*$", "", text, flags=re.M)
    # En-têtes / pieds de page répétés (« FR JO L du 12.7.2024 », « 48/144 ELI: http… »)
    text = re.sub(r"^\s*(?:FR\s+JO L du [\d.]+|JO L du [\d.]+\s+FR)\s*$", "", text, flags=re.M)
    text = re.sub(r"^.*ELI: https?://\S+.*$", "", text, flags=re.M)
    text = RE_FOOTNOTE.sub("", text)
    # Fins de notes de bas de page coupées sur plusieurs lignes (« 12.7.2024, p. 1). »)
    text = re.sub(r"^(?:.*\(JO L[^)]*|[\d.]*,? ?p\. \d+\)?[.;]?|.*\(JO L.*p\. \d+\)[.;]?)$", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------------
# 2. Découpage en unités juridiques
# ----------------------------------------------------------------------------
# Les titres sont seuls sur leur ligne dans le texte officiel ; c'est ce qui évite
# de confondre « Article 5 » (titre) et « l'article 5 » (renvoi au fil du texte).
RE_ARTICLE = re.compile(r"^(?:Article|ARTICLE)\s+(\d+|premier)\s*$", re.M)
RE_ANNEX = re.compile(r"^(?:ANNEXE|ANNEX)\s+([IVX]+)\s*$", re.M)
RE_CHAPTER = re.compile(r"^(?:CHAPITRE|CHAPTER)\s+([IVX]+)\s*$", re.M)
RE_RECITAL = re.compile(r"^\((\d{1,3})\)\s+(?=\S)", re.M)
# Notes de bas de page du JO (« (4) JO C 517… », « (7) Règlement (CE) n° … ») : même forme
# que les considérants « (4) Le… », mais elles commencent par une référence à un acte.
RE_FOOTNOTE = re.compile(
    r"^\(\d{1,3}\)\s+(?:JO\b|Règlement\b|Directive\b|Décision\b|Décision-cadre\b|Position\b|"
    r"Recommandation\b|Résolution\b|Conseil\b|Convention\b|Communication\b|Avis\b|Arrêt\b).*$",
    re.M,
)


def _title_and_body(block: str) -> tuple[str, str]:
    """Le premier paragraphe non vide après « Article N » est l'intitulé."""
    lines = [l.strip() for l in block.strip().splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return "", ""
    # Un intitulé ne commence pas par « 1. » ni « (a) » et reste court.
    first = lines[0]
    if len(first) <= 160 and not re.match(r"^(\d+\.|\([a-z0-9]+\))", first):
        return first, "\n".join(lines[1:])
    return "", "\n".join(lines)


def parse_units(raw: str) -> list[Unit]:
    text = clean(raw)

    # Repères de chapitres (position -> libellé), utilisés pour annoter les articles.
    chapters = [(m.start(), f"CHAPITRE {m.group(1)}") for m in RE_CHAPTER.finditer(text)]

    def chapter_at(pos: int) -> str:
        current = ""
        for start, label in chapters:
            if start <= pos:
                current = label
            else:
                break
        return current

    art_matches = list(RE_ARTICLE.finditer(text))
    annex_matches = list(RE_ANNEX.finditer(text))
    first_article = art_matches[0].start() if art_matches else len(text)

    # --- Considérants : tout ce qui précède l'article 1 -------------------------
    units: list[Unit] = []
    preamble = text[:first_article]
    # Un considérant n'est retenu que s'il suit la numérotation : 1, 2, 3… (filtre les
    # numérotations parasites restantes).
    rec_matches, expected = [], 1
    for m in RE_RECITAL.finditer(preamble):
        if int(m.group(1)) == expected:
            rec_matches.append(m)
            expected += 1
    for i, m in enumerate(rec_matches):
        end = rec_matches[i + 1].start() if i + 1 < len(rec_matches) else len(preamble)
        body = re.split(r"\n\s*ONT ADOPTÉ LE PRÉSENT RÈGLEMENT", preamble[m.end():end])[0].strip()
        if body:
            units.append(Unit("recital", m.group(1), "", "", body))

    # --- Articles & annexes -----------------------------------------------------
    boundaries = sorted(
        [(m.start(), m.end(), "article", "1" if m.group(1) == "premier" else m.group(1)) for m in art_matches]
        + [(m.start(), m.end(), "annex", m.group(1)) for m in annex_matches]
        # Les titres de chapitre coupent aussi le dernier article du chapitre précédent.
        + [(m.start(), m.end(), "chapter", m.group(1)) for m in RE_CHAPTER.finditer(text)]
    )
    found: dict[tuple[str, str], Unit] = {}
    for i, (start, end, kind, num) in enumerate(boundaries):
        if kind == "chapter":
            continue
        stop = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        title, body = _title_and_body(text[end:stop])
        if not body:
            continue
        body = re.split(r"\nFait à Bruxelles, le", body)[0].strip()   # signatures (art. 113)
        unit = Unit(kind, num, title, chapter_at(start) if kind == "article" else "", body)
        key = (kind, num)
        # Si une table des matières crée des doublons, on garde la version la plus longue.
        if key not in found or len(unit.text) > len(found[key].text):
            found[key] = unit

    def sort_key(u: Unit):
        return (0 if u.kind == "article" else 1, int(u.number) if u.number.isdigit() else _roman(u.number))

    units.extend(sorted(found.values(), key=sort_key))
    return units


def _roman(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        total += -vals[a] if vals.get(b, 0) > vals[a] else vals[a]
    return total


# ----------------------------------------------------------------------------
# 3. Sauvegarde / chargement
# ----------------------------------------------------------------------------
def save_units(units: list[Unit], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps({**asdict(u), "label": u.label}, ensure_ascii=False) + "\n")


def load_units(path: Path) -> list[Unit]:
    units = []
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        d.pop("label", None)
        units.append(Unit(**d))
    return units


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.exit("Usage : python -m src.ingest <fichier .pdf|.html|.txt>")
    units = parse_units(load_text(Path(argv[1])))
    out = Path("data/processed/units.jsonl")
    save_units(units, out)
    n = {k: sum(u.kind == k for u in units) for k in ("recital", "article", "annex")}
    print(f"{len(units)} unités -> {out}  {n}")
    # Contrôle de cohérence : l'AI Act compte 113 articles, 13 annexes, 180 considérants.
    expected = {"recital": 180, "article": 113, "annex": 13}
    for k, v in expected.items():
        if n[k] != v:
            print(f"  ⚠ {k}: {n[k]} trouvés, {v} attendus : vérifier l'extraction")


if __name__ == "__main__":
    main(sys.argv)
