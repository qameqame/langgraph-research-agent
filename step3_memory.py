"""
Step 3: メモリ永続化(会話を継続する)
======================================
Step2までは graph.invoke() を呼ぶたびに状態がリセットされていた。
ここでは Checkpointer を追加し、thread_id ごとに状態を保存・復元できるようにする。

これにより:
    「レポート作って」 -> 「もっと短くして」
のような複数ターンのやり取りが、直前の文脈を保ったまま可能になる。

本番運用ではSqliteSaver/PostgresSaverなどの永続ストレージに差し替え可能。
ここでは学習用にインメモリのMemorySaverを使う。
"""

from datetime import date
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

MEMBERS = ["Researcher", "Writer"]
llm = ChatOllama(model="qwen3:30b", temperature=0)

# ローカルLLM(Ollama)は今の日付を知らないため、明示的に伝えておく
# (伝えないと「学習データの頃が現在」だと誤解し、検索すべき場面で検索しない等の
# 誤判断をすることがある)。
TODAY_NOTE = (
    f"今日の日付は{date.today().isoformat()}です。"
    "あなたの学習データの知識は古い可能性があるため、"
    "最新情報や特定の年に関する質問には自分の知識だけで判断せず、"
    "必要に応じて検索ツールを使って確認してください。"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}
判断基準:
- まだ十分な情報が集まっていない場合は Researcher
- 情報は揃っていてレポートがまだ無い/修正が必要な場合は Writer
- レポートが完成し、これ以上作業が不要な場合は FINISH
※ユーザーからの追加依頼(短くして、等)がある場合もWriterに再依頼すること。
"""


def supervisor_node(state: State) -> State:
    messages = [("system", SUPERVISOR_PROMPT)] + state["messages"]
    decision = llm.with_structured_output(RouteDecision).invoke(messages)
    return {"next": decision.next}


search_tool = TavilySearchResults(max_results=3)
researcher_agent = create_react_agent(
    llm, tools=[search_tool],
    prompt=TODAY_NOTE + "\n\n"
           "あなたはリサーチ専門エージェントです。検索ツールで事実情報を集め、箇条書きで報告してください。",
)


def researcher_node(state: State) -> State:
    result = researcher_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]
    return {"messages": [("ai", f"[Researcher]\n{last.content}")]}


def writer_node(state: State) -> State:
    prompt = [
        ("system", TODAY_NOTE + "\n\n"
                   "あなたはレポート執筆の専門エージェントです。会話履歴(過去のリサーチ結果や、"
                   "ユーザーからの修正依頼)を踏まえてレポートを作成・修正してください。"),
        *state["messages"],
    ]
    response = llm.invoke(prompt)
    return {"messages": [("ai", f"[Writer]\n{response.content}")]}


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"Researcher": "Researcher", "Writer": "Writer", "FINISH": END},
)
graph_builder.add_edge("Researcher", "supervisor")
graph_builder.add_edge("Writer", "supervisor")

# --- ここがStep2との差分: checkpointerを渡してcompile ---
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    # thread_idが同じなら会話が継続される(別スレッドなら完全に別文脈になる)
    config = {"configurable": {"thread_id": "demo-thread-1"}, "recursion_limit": 25}

    print("=== ターン1: レポート作成依頼 ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについてレポートを作って")]},
        config,
    )
    print(result["messages"][-1].content)

    print("\n=== ターン2: 修正依頼(前の文脈を引き継ぐ) ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="3行程度に要約して")]},
        config,
    )
    print(result["messages"][-1].content)
