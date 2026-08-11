"""
Step 1: 単一エージェント + 検索ツール
======================================
LangGraphの最も基本的な構成を学ぶ。

- State: グラフ全体で共有するデータ構造(ここではメッセージ履歴のみ)
- Node: 処理の単位(agentノード、toolノード)
- Edge: ノード間の遷移。ここでは「ツールを呼ぶか、終了するか」を条件分岐で決める

構成:
    START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

実行:
    python step1_single_agent.py
"""

import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# --- 1. State定義 ---
# add_messages: 新しいメッセージを既存リストに"追記"していく特殊な集約関数
class State(TypedDict):
    messages: Annotated[list, add_messages]


# --- 2. ツールとLLMの準備 ---
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# --- 3. ノード定義 ---
def agent_node(state: State) -> State:
    """LLMを呼び出し、必要ならツール呼び出しを含む応答を生成する"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# --- 4. グラフの組み立て ---
graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "agent")
# tools_condition: 直前のAIメッセージにtool_callsがあれば"tools"へ、なければENDへ分岐
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


if __name__ == "__main__":
    question = "2025年から2026年にかけてのLangGraphの主要なアップデートを教えて"
    result = graph.invoke({"messages": [("user", question)]})

    print("=== 最終回答 ===")
    print(result["messages"][-1].content)

    print("\n=== 会話の全ステップ ===")
    for m in result["messages"]:
        print(f"[{m.type}] {str(m.content)[:200]}")
