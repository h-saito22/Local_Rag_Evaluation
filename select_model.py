import json
from pathlib import Path
with open(Path(__file__).resolve().parents[0] / "select_model.json", "r", encoding="utf-8") as f:
    d = json.load(f)

select = d[d["switch"]]
embeddings = select["Embeddings"]
LLM = select["LLM"]

print(f"Embeddings:{embeddings}",
      f"LLM : {LLM}" )



