import json
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama ,OllamaEmbeddings

with open(Path(__file__).resolve().parents[0] / "select_model.json", "r", encoding="utf-8") as f:
    d = json.load(f)

select = d[d["switch"]]
select_dic = d["switch"]
embeddings = select["Embeddings"]
embeddings_class = select["Embeddings_class"]
LLM = select["LLM"]
LLM_class = select["LLM_class"]
temperature = d["temperature"]


if LLM_class == "ChatGoogleGenerativeAI":
    select_model = ChatGoogleGenerativeAI(model=LLM,temperature=temperature)
elif LLM_class == "ChatOllama":
    select_model = ChatOllama(model=LLM,temperature=temperature)
else :raise ValueError(f"{LLM_class}は存在しないです。使えるモデルは[ChatGoogleGenerativeAI,ChatOllama]です。選択箇所は{select_dic}になっています。")


if embeddings_class == "GoogleGenerativeAIEmbeddings":
    embedding_model = GoogleGenerativeAIEmbeddings(model = embeddings)
elif embeddings_class == "OllamaEmbeddings":
    embedding_model = OllamaEmbeddings(model =embeddings)
else : raise ValueError(f"埋め込みモデルが違います。選択した埋め込みモデル{embeddings}")