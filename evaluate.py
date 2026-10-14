from  pathlib import Path
import csv
from lg2_rag_graph import app 
import numpy as np
import time

csv_path = Path(__file__).resolve().parents[0] / "data" / "qa" / "qa_v1.csv"
with open(csv_path ,encoding="utf-8" , newline="") as f :
    questions = csv.DictReader(f)
    test = list(questions)

rank_list = []
time_list = []
for init_question in test[:5]:
    if init_question["type"] == "none":
        continue
    question= init_question["question"]
    source_file = init_question["source_file"]
    start_time = time.perf_counter()
    result = app.invoke({"question" : question})
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(init_question)
    print(result)
    print(total_time)
    time_list.append(total_time)
    
    for i, chunk in enumerate(result["retrieved"]):
        if init_question["evidence"]  in chunk["content"]:
            rank = i + 1
            rank_list.append(rank)
            break
        elif len(result["retrieved"]) - 1 ==  i:
            rank_list.append(None)

hit_list = [r for r in rank_list if r is not None]
hit_at_2 = len(hit_list) / len(rank_list)
mrr = sum(1 / r for r in hit_list) / len(rank_list)
print("hit@2:", hit_at_2, "MRR:", mrr)
print(rank_list)