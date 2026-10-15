from pathlib import Path
import csv


csv_path = Path(__file__).resolve().parents[0] / "data" / "qa" / "qa_v1.csv"
with open(csv_path ,encoding="utf-8" , newline="") as f :
    questions = csv.DictReader(f)
    question_answer = list(questions)

csv_path = Path(__file__).resolve().parents[0] / "results"  / "2026_10_15_qwen3-8b.csv"
with open(csv_path ,encoding="utf-8" , newline="") as f :
    results = csv.DictReader(f)
    result = list(results)

qa_by_id = {row["id"]: row for row in question_answer}

NOT_FOUND = "資料に記載がありません"
def judge(qa, answer):
    if qa["type"] == "none":
        return answer.rstrip("。") == NOT_FOUND
    if NOT_FOUND in answer:
        return False
    keywords = qa["keywords"].split("|")
    return any(k in answer for k in keywords)

scores = {}
for r in result:
    qa = qa_by_id[r["id"]]
    ok = judge(qa, r["answer"])
    scores.setdefault(qa["type"], []).append(ok)

for t , oks in scores.items():
    print(t, sum(oks) , "/" , len(oks) , "=" , sum(oks) / len(oks))

