
import os
from pathlib import Path
import csv
import random 
import json

random.seed(42)
train_dir = Path(__file__).resolve().parents[0] /"data" / "lora" / "docs"

train_text_files = list(train_dir.rglob("*.md"))
train_text_files = sorted(train_text_files)

content_list = []
section_dict = {}
print(train_text_files)
train_text_list = []
for train_text in train_text_files :
    train_text_name = Path(train_text).name
    content = train_text.read_text(encoding="utf-8")
    contents = content.split("## ")
    for text_contents in contents[1:]:
        heading = text_contents.split("\n")[0]
        section_dict[(train_text_name,heading)] = ("## " + text_contents).strip()

csv_train_path = Path(__file__).resolve().parents[0] / "data" / "lora" / "train_qa.csv"
with open(csv_train_path ,encoding="utf-8" , newline="") as f :
    questions = csv.DictReader(f)
    train_questions = list(questions)

records_by_type = {}
for row  in train_questions:
    source_file = row["source_file"]
    section = row["section"]
    answer_chunk= []
    if section != "":
        for s in section.split("|"):
            answer_chunk.append(section_dict[(source_file , s)])

    candidates = [c for c in section_dict.values() if c not in answer_chunk]

    picked =random.sample(candidates,2 - len(answer_chunk))
    answer_chunk.extend(picked)
    random.shuffle(answer_chunk)
    
    context = "\n---\n".join(answer_chunk)

    prompt = f"""
        以下の資料だけを根拠に答えてください。
    資料の中に、質問と同じ意味の内容がある場合は、言い回しが違っても、その内容を使って答えてください。
    質問に関係する内容が資料にない場合に限り、「資料に記載がありません」とだけ返答してください。

        #資料
        {context}
        #質問
        {row["question"]}
        """

    record = {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["answer"]},
    ]}
    records_by_type.setdefault(row["type"], []).append(record)

train ,valid= [] , []
for t ,recs in records_by_type.items():
    random.shuffle(recs)
    n = int(len(recs) * 0.8)
    train.extend(recs[:n])
    valid.extend(recs[n:])
random.shuffle(train)

out_dir = Path(__file__).resolve().parents[0] / "data" / "lora"
with open(out_dir / "train.jsonl" , "w", encoding="utf-8") as f :
    for record in train:
        f.write(json.dumps(record,ensure_ascii=False) + "\n")
with open(out_dir / "valid.jsonl" , "w", encoding="utf-8") as f :
    for record in valid:
        f.write(json.dumps(record,ensure_ascii=False) + "\n")