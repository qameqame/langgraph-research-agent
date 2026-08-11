"""
解答: 課題1-1 計算ツールの追加
======================================
"""

import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def calculator(expression: str) -> str:
    """四則演算の式(例: "2026 - 2016")を計算して結果を文字列で返す。
    加減乗除・括弧のみに対応。
    """
    # 安全のため、数字・演算子・空白・括弧・小数点のみを許可してからevalする
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return f"エラー: 使用できない文字が含まれています: {expression}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"


search_tool = TavilySearchResults(max_results=3)
tools = [search_tool, calculator]

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
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
        for m in result["messages"]:
            if getattr(m, "tool_calls", None):
                print(f"  -> 呼ばれたツール: {[tc['name'] for tc in m.tool_calls]}")

# 解説:
# - @tool デコレータを付けた関数は、関数名がツール名、docstringがLLMへの説明文になる。
#   型ヒント(expression: str)は引数スキーマとしてそのままLLMに渡される。
# - ToolNode(tools) は tools リストに入っている全ツールをname引きできるように保持し、
#   AIMessage.tool_calls の name フィールドを見て対応する関数を呼び出す。
#   ツールを追加するだけで、ToolNode側のコード変更は不要な設計になっている。
# - 1問目は計算ツール、2問目は検索ツールが呼ばれることを確認できるはず。
#   (LLMの判断なので毎回100%同じとは限らない点に注意)
