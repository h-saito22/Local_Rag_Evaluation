import ollama

question = ["有給休暇は3日前までに申請してください","休みを取るときは、3日前に届け出が必要です"]
response = ollama.embed(
    model='qwen3-embedding:0.6b',
    input=question,
)
print(len(response.embeddings[0]))