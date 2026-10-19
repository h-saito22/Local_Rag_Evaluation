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
import time

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
metadata_list = []
h = hashlib.sha256()
for text in text_files :
    text_name = Path(text).name
    text = text.read_text(encoding="utf-8")
    h.update(text.encode("utf-8"))
    text = splitter.split_text(text)
    metadata_list.extend([{"source": text_name}] * len(text))
    text_list.extend(text)
    
print(h.hexdigest())
print(len(text_list), len(metadata_list))
models_metadata.update({"hash recoginition" : h.hexdigest(),
                                          "chunk_size" : CHUNK_SIZE,
                                          "chunk_overlap" : CHUNK_OVERLAP})

embenddings = embedding_model
store = Chroma(
embedding_function=embedding_model,
    persist_directory=Path(__file__).resolve().parents[0] / "chroma_db/"
)

data = store.get()
if len(data["documents"]) == 0 :
    store.add_texts(text_list, metadatas=metadata_list)
    print("インデックスを作成しました。")
    store._collection.modify(metadata = models_metadata ) #models_metadata 
    print(store._collection.metadata)
else:
    if store._collection.metadata is None :
        raise ValueError("chroma_dbのインデックスに、作成時の設定メモ（メタデータ）がありません。\n"
                         "chroma_dbフォルダを削除して実行すると、インデックスが作り直されます。")
    for k in models_metadata:
        if k not in store._collection.metadata :
              raise ValueError(f"対象のインデックスが古いのでchroma_dbを削除してから再実行してください。")
      
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
        raise ValueError(f"chroma_dbのインデックスで、チャンクサイズ（CHUNK_SIZE）の食い違いがありました。\n"
                         f"このインデックスは、チャンクサイズ{stored_chunk_size}で作成されましたが現在は、{current_chunk_size}です。\n"
                         f"保存されたインデックスを使用する際は、lg2_rag_graph.py内のCHUNK_SIZEを{stored_chunk_size}に変更して再実行してください。\n"
                         f"現在の設定で行う場合は、chroma_dbフォルダを削除して実行をするとインデックスが作り直されます。")
    
    elif store._collection.metadata["chunk_overlap"] != models_metadata["chunk_overlap"]:
        stored_chunk_overlap = store._collection.metadata["chunk_overlap"]
        current_chunk_overlap = models_metadata["chunk_overlap"]
        raise ValueError(f"chroma_dbのインデックスで、チャンクオーバーラップ（CHUNK_OVERLAP）の食い違いがありました。\n"
                        f"このインデックスは、チャンクオーバーラップ{stored_chunk_overlap}で作成されましたが現在は、{current_chunk_overlap}です。\n"
                        f"保存されたインデックスを使用する際は、lg2_rag_graph.py内のCHUNK_OVERLAPを{stored_chunk_overlap}に変更して再実行してください。\n"
                        f"現在の設定で行う場合は、chroma_dbフォルダを削除して実行をするとインデックスが作り直されます。")
    
    elif store._collection.metadata["hash recoginition"] != models_metadata["hash recoginition"]:
        raise ValueError(f"chroma_db作成時の文章ファイルに食い違いがありました。\n"
                         f"文章変更に思いあたりがある場合は、chroma_dbフォルダを削除して再実行してください。\n"
                         f"思いあたりがない場合は、ターミナル上で git statusを実行して変更されたファイルを確認して修正を行ってください。")
    
    else :
        print("読み込みました。")

llm = select_model
class State(TypedDict):
    question : str 
    context : str 
    answer : str
    retrieved : list[dict] 
    retrieve_time : float
    generate_time : float
    eval_count : int 

# ---- Node: retrieve ----
def retrieve(state : State):
    print("[recive]受け取ったState", state)
    question = state["question"]
    start_time = time.perf_counter()
    result = store.similarity_search_with_score(question,k=2)
    end_time = time.perf_counter()
    retrieve_time = end_time - start_time
    retreive_list = []
    message_list = []
    for doc , score in result :
        message = doc.page_content
        message_list.append(message)
        retreive_list.append(
            {"content" : doc.page_content , 
             "source" : doc.metadata["source"] ,
             "score" : score}
        )
    context = "\n---\n".join(message_list)
    return {"context":context ,"retrieved" : retreive_list , "retrieve_time" : retrieve_time}

# ---- Node: generate ----
def generate(state:State):
    print("[generate]受け取ったState",state)
    context = state["context"]
    question = state["question"]
    


#以下の資料だけを根拠に答えてください。資料にない場合は「資料に記載がありません」と返答してください。
    prompt = f"""
    以下の資料だけを根拠に答えてください。
資料の中に、質問と同じ意味の内容がある場合は、言い回しが違っても、その内容を使って答えてください。
質問に関係する内容が資料にない場合に限り、「資料に記載がありません」とだけ返答してください。

      #資料
      {context}
     #質問
     {question}
    """
    # start_time = time.perf_counter()
    # answer = llm.invoke(prompt).text
    # end_time = time.perf_counter()
    # generate_time = end_time - start_time
    start_time = time.perf_counter()
    response = llm.invoke(prompt)          # メッセージ全体
    end_time = time.perf_counter()
    answer = response.text                 # 回答の文字列（今までどおり）
    print(response.response_metadata)      # ここにトークン数などが入っている
    eval_count = response.usage_metadata["output_tokens"]
    generate_time = end_time - start_time

    return {"answer":answer, "generate_time" : generate_time , "eval_count":eval_count}


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
