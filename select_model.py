import json


model_switch = {
    "embeddings": [ "Local", "Cloud"] ,
    "LLM" : [ "Local", "Cloud"]
}
print(model_switch)

model_switch_json = json.loads(model_switch)
