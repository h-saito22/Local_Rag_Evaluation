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


MODEL = os.getenv("GEMINI_MODEL")
text = (Path(__file__).resolve().parents[0] /"sample_doc.txt").read_text(encoding="utf-8")


# ---- 準備（ex6 の [2]〜[4] と同じ）----
# TODO: splitter → text_list → embeddings → store、llm も作っておく
splitter = RecursiveCharacterTextSplitter(chunk_size = 50 ,chunk_overlap = 20)
text_list = splitter.split_text(text)
embenddings = embedding_model
store = Chroma(
    embedding_function=embedding_model,
    persist_directory=Path(__file__).resolve().parents[0] / "chroma_db/"
)
data = store.get()
if len(data["documents"]) == 0 :
    store.add_texts(text_list)
    print("インデックスを作成しました。")
    store._collection.modify(metadata = embeddings_dic)
else:
    if store._collection.metadata["Embeddings_name"] != embeddings_dic["Embeddings_name"] :
        stored_embeddings_name = store._collection.metadata["Embeddings_name"]
        current_embeddings_name = embeddings_dic["Embeddings_name"]
        raise ValueError(f"chroma_dbのインデックスで、埋め込みモデルの食い違いがありました。\n"
                          f"このインデックスは、{stored_embeddings_name}で作成されましたが現在は、{current_embeddings_name}です。\n"
                          f"保存されたインデックスを使う場合は、select_model.jsonのモデルを、{stored_embeddings_name}に戻してください。\n"
                          f"現在の設定で行う場合は、chroma_dbフォルダを削除して実行すると、インデックスが作り直されます。")
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
    