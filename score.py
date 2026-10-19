from pathlib import Path
import csv



NOT_FOUND = "資料に記載がありません"
def judge(qa, answer):
    if qa["type"] == "none":
        return answer.strip().rstrip("。") == NOT_FOUND
    if NOT_FOUND in answer:
        return False
    keywords = qa["keywords"].split("|")
    return any(k in answer for k in keywords)


def score_file(path,qa_path):
    with open(qa_path ,encoding="utf-8" , newline="") as f :
        questions = csv.DictReader(f)
        question_answer = list(questions)

    with open(path ,encoding="utf-8" , newline="") as f :
        results = csv.DictReader(f)
        result = list(results)

    qa_by_id = {row["id"]: row for row in question_answer}

    scores = {}
    for r in result:
        qa = qa_by_id[r["id"]]
        ok = judge(qa, r["answer"])
        scores.setdefault(qa["type"], []).append(ok)

    for t , oks in scores.items():
        print(t, sum(oks) , "/" , len(oks) , "=" , sum(oks) / len(oks))

    return scores

if __name__ == "__main__":
    score_file(Path(__file__).resolve().parents[0] / "results" / "2026_10_16_qwen3-4b-q4_K_M_prompt_3b.csv",
               Path(__file__).resolve().parents[0] / "data" / "qa" / "qa_v1.csv")