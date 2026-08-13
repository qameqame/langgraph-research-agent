"""
課題1-1: 検索ツールに加えて計算ツールを追加する
======================================
Step1(step1_single_agent.py)では検索ツールのみを`bind_tools`していました。
この課題では、四則演算を行う`calculator`ツールを追加し、
エージェントが質問内容に応じて「検索」と「計算」を使い分けられるようにします。

例えば「2026年から2016年は何年経った?」という質問には計算ツールが、
「LangGraphの最新アップデートは?」という質問には検索ツールが使われるはずです。

参考ドキュメント:
- カスタムツールの作り方: https://docs.langchain.com/oss/python/langchain/tools
- ToolNodeが複数ツールをどう扱うか: https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode

進め方:
1. 下のTODOを埋める
2. `python exercises/step1_ex1_calculator_tool.py` を実行
3. うまく動いたら `exercises/answers/step1_ex1_calculator_tool.py` と見比べる
"""

import os
from datetime import date
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ローカルLLM(Ollama)は今の日付を知らないため、明示的に伝えておく
# (伝えないと「学習データの頃が現在」だと誤解し、検索すべき場面で検索しない等の
# 誤判断をすることがある)。
SYSTEM_PROMPT = (
    f"今日の日付は{date.today().isoformat()}です。"
    "あなたの学習データの知識は古い可能性があるため、"
    "最新情報や特定の年に関する質問には自分の知識だけで判断せず、"
    "必要に応じて検索ツールを使って確認してください。"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


# --- TODO 1: 計算ツールを実装する -------------------------------------
# ヒント: langchain_core.tools の @tool デコレータを使うと、
#         普通のPython関数をLangChainツールに変換できます。
#         docstring がそのままLLMへの「このツールの説明」になる点に注意。
#
# @tool
# def calculator(expression: str) -> str:
#     """四則演算の式(例: "2026 - 2016")を計算して結果を文字列で返す。"""
#     ...  # ここにevalなどを使った計算処理を書く(evalの安全性にも触れてみましょう)

# --- TODO 1 ここまで ----------------------------------------------------


search_tool = TavilySearchResults(max_results=3)

# --- TODO 2: toolsリストにcalculatorを追加する --------------------------
tools = [search_tool]  # ← ここに calculator を追加する
# --- TODO 2 ここまで ----------------------------------------------------

llm = ChatOllama(model="qwen3:30b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: State) -> State:
    messages = [("system", SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


if __name__ == "__main__":
    questions = [
        "2026年から2016年で何年経ちましたか?計算してください",
        "LangGraphの2026年時点での主要なアップデートを教えて",
    ]
    for q in questions:
        print(f"\n=== 質問: {q} ===")
        result = graph.invoke({"messages": [("user", q)]})
        print(result["messages"][-1].content)

        # どのツールが呼ばれたか確認してみる
        for m in result["messages"]:
            if getattr(m, "tool_calls", None):
                print(f"  -> 呼ばれたツール: {[tc['name'] for tc in m.tool_calls]}")
