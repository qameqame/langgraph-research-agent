"""
解答: 課題2-1 Supervisorの3方向分岐(FactChecker追加)
======================================
"""

from datetime import date
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

MEMBERS = ["Researcher", "Writer", "FactChecker"]

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
    next: Literal["Researcher", "Writer", "FactChecker", "FINISH"]


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}

判断基準:
- まだ十分な情報が集まっていない場合は Researcher
- 情報は揃っていてレポートがまだ無い場合は Writer
- レポートは書けたが、数値や固有名詞の裏取りがまだの場合は FactChecker
- FactCheckerの検証を通過したレポートがあれば FINISH
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
    return {"messages": [("ai", f"[Researcher]\n{result['messages'][-1].content}")]}


def writer_node(state: State) -> State:
    prompt = [
        ("system", TODAY_NOTE + "\n\n"
                   "あなたはレポート執筆の専門エージェントです。会話履歴のリサーチ結果を元に、"
                   "簡潔で読みやすい日本語レポートを作成してください。"),
        *state["messages"],
    ]
    response = llm.invoke(prompt)
    return {"messages": [("ai", f"[Writer]\n{response.content}")]}


fact_checker_agent = create_react_agent(
    llm, tools=[search_tool],
    prompt=(
        TODAY_NOTE + "\n\n"
        "あなたはファクトチェック専門エージェントです。直前のWriterのレポートに含まれる"
        "数値・統計・固有名詞などを検索して裏取りしてください。"
        "問題があれば具体的に指摘し、問題が無ければ『検証OK: 主要な主張は裏付けが取れました』"
        "とだけ報告してください。"
    ),
)


def fact_checker_node(state: State) -> State:
    result = fact_checker_agent.invoke({"messages": state["messages"]})
    return {"messages": [("ai", f"[FactChecker]\n{result['messages'][-1].content}")]}


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)
graph_builder.add_node("FactChecker", fact_checker_node)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FactChecker": "FactChecker",
        "FINISH": END,
    },
)
graph_builder.add_edge("Researcher", "supervisor")
graph_builder.add_edge("Writer", "supervisor")
graph_builder.add_edge("FactChecker", "supervisor")

graph = graph_builder.compile()


if __name__ == "__main__":
    topic = "生成AIエージェントの2026年時点でのビジネス活用トレンド"
    result = graph.invoke(
        {"messages": [HumanMessage(content=f"次のテーマでレポートを作成してください: {topic}")]},
        {"recursion_limit": 30},
    )
    print("=== 最終出力 ===")
    print(result["messages"][-1].content)

# 解説:
# - MEMBERS / RouteDecision.next の両方にFactCheckerを追加しないと、
#   Supervisorがルーティング先として選べない(構造化出力のLiteralに無い値は
#   そもそも生成され得ない)ため、両方の変更が必須になる点がポイント。
# - FactCheckerもResearcherと同じくcreate_react_agentで実装できる。
#   「役割ごとにprompt(役割の定義)を変えるだけで、同じ部品(検索ツール+ReAct)を
#   使い回せる」のがマルチエージェント設計の利点。
# - SUPERVISOR_PROMPTの判断基準にFactCheckerのタイミングを明記しないと、
#   Supervisorが「Writerの後は毎回FINISH」のように誤った判断をしがちなので、
#   プロンプト側の記述も忘れずに更新する必要がある。
