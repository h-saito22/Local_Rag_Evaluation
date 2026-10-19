from  pathlib import Path
import csv
from lg2_rag_graph import app 
import numpy as np
import time
from score import score_file , judge

qa_path = Path(__file__).resolve().parents[0] / "data" / "qa" / "qa_v1.csv"

# LoRA用のデータ
# qa_path = Path(__file__).resolve().parents[0] / "data" / "qa" / "qa_hard.csv"
with open(qa_path ,encoding="utf-8" , newline="") as f :
    questions = csv.DictReader(f)
    test = list(questions)
result_list = []
rank_list = []
time_list = []
print(test)
for init_question in test:
    question= init_question["question"]
    source_file = init_question["source_file"]
    start_time = time.perf_counter()
    result = app.invoke({"question" : question})
    end_time = time.perf_counter()
    total_time = end_time - start_time

    rank = None
    if init_question["type"] != "none":
        for i, chunk in enumerate(result["retrieved"]):
            if init_question["evidence"]  in chunk["content"]:
                rank = i + 1
                break
        rank_list.append(rank)

    result_list.append({
        "id" : init_question["id"],
        "type" : init_question["type"],
        "rank" : rank,
        "retrieve_time" : result["retrieve_time"],
        "generate_time" : result["generate_time"],
        "answer" : result["answer"],
        "eval_count" : result["eval_count"],
    })
        

hit_list = [r for r in rank_list if r is not None]
hit_at_2 = len(hit_list) / len(rank_list)
mrr = sum(1 / r for r in hit_list) / len(rank_list)

save_csv_dir = Path(__file__).resolve().parents[0] / "results"
save_csv_dir.mkdir(exist_ok=True)
save_path = save_csv_dir /"2026_10_19_qwen3-4b-mlx-8bit_prompt_3b_qa_v1_LoRA.csv"

with open(save_path,"w",encoding="utf-8" , newline="") as f:
    writer =csv.DictWriter(f ,fieldnames=result_list[0].keys())
    writer.writeheader()
    writer.writerows(result_list)

score_dic = score_file(save_path,qa_path)