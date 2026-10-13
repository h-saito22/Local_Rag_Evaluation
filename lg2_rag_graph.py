"""
LangGraph 練習2: Step1 の RAG を LangGraph で書く

    START → [retrieve] → [generate] → END

ex4 との対応:
    search()                 → retrieve ノード（質問で検索し、context を State に書く）
    build_prompt() + answer() → generate ノード（context と question から answer を State に書く）

State に必要なキーを考えよう:
    入力: question / 途中: context / 出力: answer
"""
import os
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
load_dotenv(Path(__file__).resolve().parents[0] / ".env")
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from select_model import select_model ,embeddings_dic , embedding_model
from langchain_chroma import Chroma
import hashlib

CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
models_metadata = embeddings_dic.copy()
MODEL = os.getenv("GEMINI_MODEL")
base_dir = Path(__file__).resolve().parents[0] /"data" / "docs"
text_files = list(base_dir.rglob("*.md"))
text_files = sorted(text_files)
print(text_files)
splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE ,chunk_overlap = CHUNK_OVERLAP)
text_list = []
h = hashlib.sha256()
for text in text_files :
    text = text.read_text(encoding="utf-8")
    h.update(text.encode("utf-8"))
    text = splitter.split_text(text)
    text_list.extend(text)
    
print(h.hexdigest())
models_metadata.update({"hash recoginition" : h.hexdigest(),
                                          "chunk_size" : CHUNK_SIZE,
                                          "chunk_overlap" : CHUNK_OVERLAP})
print(models_metadata)
embenddings = embedding_model
store = Chroma(
embedding_function=embedding_model,
    persist_directory=Path(__file__).resolve().parents[0] / "chroma_db/"
)

data = store.get()
if len(data["documents"]) == 0 :
    store.add_texts(text_list)
    print("インデックスを作成しました。")
    store._collection.modify(metadata = models_metadata)
    print(store._collection.metadata)
else:
    if store._collection.metadata["Embeddings_name"] != models_metadata["Embeddings_name"] :
        stored_embeddings_name = store._collection.metadata["Embeddings_name"]
        current_embeddings_name = models_metadata["Embeddings_name"]
        raise ValueError(f"chroma_dbのインデックスで、埋め込みモデルの食い違いがありました。\n"
                          f"このインデックスは、{stored_embeddings_name}で作成されましたが現在は、{current_embeddings_name}です。\n"
                          f"保存されたインデックスを使う場合は、select_model.jsonのモデルを、{stored_embeddings_name}に戻してください。\n"
                          f"現在の設定で行う場合は、chroma_dbフォルダを削除して実行すると、インデックスが作り直されます。")
    elif store._collection.metadata["chunk_size"] != models_metadata["chunk_size"]:
        stored_chunk_size = store._collection.metadata["chunk_size"]
        current_chunk_size = models_metadata["chunk_size"]
        raise ValueError(f"chroma_dbのインデックスで、埋め込みモデルの食い違いがありました。\n"
                         f"このインデックスは、{stored_chunk_size}で作成されましたが現在は、{current_chunk_size}です。\n"
                         f"保存されたインデックスを使用する際は、現在のチャンクサイズを{stored_chunk_size}に変更して再実行してください。\n"
                         f"現在の設定で行う場合は、chroma_dbフォルダを削除して実行をするとインデックスが作り直されます。")
    elif store._collection.metadata["chunk_overlap"] != models_metadata["chunk_overlap"]:
        stored_chunk_overlap = store._collection.metadata["chunk_overlap"]
        cuurent_chunk_overlap = models_metadata["chunk_overlap"]
        raise ValueError(f"chroma_dbのインデックスで、埋め込みモデルの食い違いがありました。\n"
                        f"このインデックスは、{stored_chunk_overlap}で作成されましたが現在は、{cuurent_chunk_overlap}です。\n"
                        f"保存されたインデックスを使用する際は、現在のチャンクサイズを{stored_chunk_overlap}に変更して再実行してください。\n"
                        f"現在の設定で行う場合は、chroma_dbフォルダを削除して実行をするとインデックスが作り直されます。")
    elif store._collection.metadata["hash recoginition"] != models_metadata["hash recoginition"]:
        raise ValueError(f"chroma_db作成時の文章ファイルに食い違いがありました。\n",
                         f"文章変更に思い当たりがある場合は,chroma_dbフォルダを削除して再実行してください。\n"
                         f"思いあたりがない場合は、ターミナル上で git stutusを実行して変更されたファイルを確認して修正を行ってください。")
    else :
        print("読み込みました。")

llm = select_model
class State(TypedDict):
    question : str 
    context : str 
    answer : str


# ---- Node: retrieve ----
def retrieve(state : State):
    print("[recive]受け取ったState", state)
    question = state["question"]

    result = store.similarity_search_with_score(question,k=2)
    message_list = []
    for doc , score in result :
        message = doc.page_content
        message_list.append(message)
    context = "\n---\n".join(message_list)
    return {"context":context}

# ---- Node: generate ----
def generate(state:State):
    print("[generate]受け取ったState",state)
    context = state["context"]
    question = state["question"]


# 資料にない場合は、資料の中から関係しそうな文章を示したうえで、資料に記載はなかったと伝えてください。
    prompt = f"""
    以下の資料だけを根拠に答えてください。資料にない場合は「資料に記載がありません」と返答してください。

      #資料
      {context}
     #質問
     {question}
    """
    answer = llm.invoke(prompt).text
    return {"answer":answer}


# ---- グラフ ----
graph = StateGraph(State)
graph.add_node("retrieve",retrieve)
graph.add_node("generate", generate)
graph.add_edge(START , "retrieve")
graph.add_edge("retrieve","generate")
graph.add_edge("generate",END)
app=graph.compile()


if __name__ == "__main__":
    q = input("質問 > ")
    result = app.invoke({"question":q})
    print("\n[回答]")
    print(result["answer"])
    print()
    print(result)
    print(F"埋め込みモデル :{embedding_model}",
          f"Chat Model : {select_model}")
    