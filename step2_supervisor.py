"""
Step 2: Supervisorパターンでマルチエージェント化
======================================
1つのLLMに全部やらせるのではなく、役割を分割する。

- Researcher: Web検索して情報を集める専門エージェント (Step1のReActエージェントを再利用)
- Writer:     集めた情報からレポートを執筆する専門エージェント
- Supervisor: 「次に誰を動かすか」を毎ターン判断する司令塔

構成:
    START -> supervisor -> (Researcher | Writer | END) -> supervisor -> ...

ポイント:
    Supervisorは通常のLLM呼び出しだが、出力を構造化(Literal型)することで
    「次のノード名」という機械可読な決定を返させる。
"""

import os
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

MEMBERS = ["Researcher", "Writer"]
OPTIONS = MEMBERS + ["FINISH"]

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


# --- State定義 ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


# --- Supervisorの出力スキーマ ---
class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
以下のメンバーと会話しながら、次にどのメンバーを動かすか判断してください。
メンバー: {MEMBERS}

判断基準:
- まだ十分な情報が集まっていない場合は Researcher
- 情報は揃っていてレポートがまだ無い/不十分な場合は Writer
- レポートが完成し、これ以上作業が不要な場合は FINISH
"""


def supervisor_node(state: State) -> State:
    messages = [("system", SUPERVISOR_PROMPT)] + state["messages"]
    decision = llm.with_structured_output(RouteDecision).invoke(messages)
    return {"next": decision.next}


# --- Researcherエージェント(Step1のReActパターンをprebuiltで再利用) ---
search_tool = TavilySearchResults(max_results=3)
researcher_agent = create_react_agent(
    llm, tools=[search_tool],
    prompt="あなたはリサーチ専門エージェントです。検索ツールを使って事実情報を集めてください。推測や執筆は行わず、集めた事実を箇条書きで報告してください。",
)


def researcher_node(state: State) -> State:
    result = researcher_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]
    return {"messages": [("ai", f"[Researcher]\n{last.content}")]}


# --- Writerエージェント(ツールなし、執筆に専念) ---
def writer_node(state: State) -> State:
    prompt = [
        ("system", "あなたはレポート執筆の専門エージェントです。これまでの会話にあるリサーチ結果を元に、"
                   "簡潔で読みやすい日本語レポート(見出し・要点付き)を作成してください。"),
        *state["messages"],
    ]
    response = llm.invoke(prompt)
    return {"messages": [("ai", f"[Writer]\n{response.content}")]}


# --- グラフ組み立て ---
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

graph = graph_builder.compile()


if __name__ == "__main__":
    topic = "生成AIエージェントの2026年時点でのビジネス活用トレンド"
    result = graph.invoke(
        {"messages": [HumanMessage(content=f"次のテーマでレポートを作成してください: {topic}")]},
        {"recursion_limit": 25},
    )

    print("=== 最終レポート ===")
    print(result["messages"][-1].content)
