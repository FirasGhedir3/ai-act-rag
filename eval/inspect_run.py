import json, sqlite3, sys

variant = sys.argv[1] if len(sys.argv) > 1 else "B"
qs = {q["id"]: q for q in map(json.loads, open("eval/questions.jsonl", encoding="utf-8"))}
con = sqlite3.connect("eval/results.sqlite")
for qid, question, answer, labels in con.execute(
        "SELECT qid, question, answer, retrieved_labels_json FROM runs WHERE variant=? ORDER BY qid", (variant,)):
    print("=" * 70)
    print(qid, question)
    print("ARTICLE ATTENDU :", qs[qid]["expected_articles"])
    print("PASSAGES TROUVÉS :", [l[0] if l else "?" for l in json.loads(labels)])
    print("RÉPONSE :", answer[:300].replace("\n", " "))