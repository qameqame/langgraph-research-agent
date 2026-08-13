"""
課題2-1: Supervisorの分岐を3方向にする(FactCheckerを追加)
======================================
Step2ではSupervisorがResearcher/Writerの2人に振り分けていました。
この課題では、Writerが書いたレポート中の数値や固有名詞を検索で裏取りする
「FactChecker」エージェントを追加し、Supervisorが3人に振り分けられるようにします。

構成イメージ:
    Supervisor --> Researcher | Writer | FactChecker | FINISH

参考ドキュメント:
- Multi-agent構成の考え方: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
- create_react_agent: https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent
- Pydanticでの構造化出力(Literalの拡張): https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic/with_structured_output

進め方:
1. 下のTODOを埋める(RouteDecisionの選択肢追加、fact_checker_nodeの実装、グラフへの追加)
2. `python exercises/step2_ex1_three_way_supervisor.py` を実行
3. `exercises/answers/step2_ex1_three_way_supervisor.py` と見比べる
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

# --- TODO 1: メンバーにFactCheckerを追加する -----------------------------
MEMBERS = ["Researcher", "Writer"]  # ← "FactChecker" を追加する
# --- TODO 1 ここまで ------------------------------------------------------

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


# --- TODO 2: RouteDecisionのLiteralにFactCheckerを追加する ----------------
class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]  # ← "FactChecker" を追加
# --- TODO 2 ここまで ------------------------------------------------------


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


# --- TODO 3: FactCheckerエージェント/ノードを実装する ---------------------
# ヒント: Researcher同様、検索ツールを持つ create_react_agent として作れます。
#         プロンプトは「Writerのレポート内の数値・固有名詞を検索で検証し、
#         問題があれば指摘、問題なければ『検証OK』と報告する」ような内容にする。
#
# fact_checker_agent = create_react_agent(
#     llm, tools=[search_tool],
#     prompt="...",
# )
#
# def fact_checker_node(state: State) -> State:
#     ...

# --- TODO 3 ここまで ------------------------------------------------------


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)
# --- TODO 4: FactCheckerノードをグラフに追加する --------------------------
# graph_builder.add_node("FactChecker", fact_checker_node)
# --- TODO 4 ここまで ------------------------------------------------------

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        # --- TODO 5: FactCheckerへの分岐を追加する ---
        # "FactChecker": "FactChecker",
        "FINISH": END,
    },
)
graph_builder.add_edge("Researcher", "supervisor")
graph_builder.add_edge("Writer", "supervisor")
# --- TODO 6: FactChecker -> supervisor のエッジを追加する ------------------
# graph_builder.add_edge("FactChecker", "supervisor")
# --- TODO 6 ここまで -------------------------------------------------------

graph = graph_builder.compile()


if __name__ == "__main__":
    topic = "生成AIエージェントの2026年時点でのビジネス活用トレンド"
    result = graph.invoke(
        {"messages": [HumanMessage(content=f"次のテーマでレポートを作成してください: {topic}")]},
        {"recursion_limit": 30},
    )
    print("=== 最終出力 ===")
    print(result["messages"][-1].content)
