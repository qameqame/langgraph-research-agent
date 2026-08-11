"""
Step 4: Criticによる自己修正ループ
======================================
Step3のグラフに「品質チェック」を担当するCriticエージェントを追加する。

構成:
    ... -> Writer -> Critic -> (差し戻し: Writer / 承認: supervisor) -> ...

ポイント:
- Criticは合格/差し戻しを構造化出力で判定し、差し戻し時はフィードバックを
  メッセージとして会話履歴に追加する(Writerが次回それを踏まえて書き直す)。
- 無限ループ防止のため revision_count に上限(MAX_REVISIONS)を設ける。
  これはLangGraphに限らず自己修正ループを作る際に必須の安全装置。
"""

from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

MEMBERS = ["Researcher", "Writer"]
MAX_REVISIONS = 2  # Critic差し戻しの上限回数

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    revision_count: int


class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str  # 差し戻す場合の具体的な改善指示


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}
判断基準:
- まだ十分な情報が集まっていない場合は Researcher
- 情報は揃っていてレポートがまだ無い場合は Writer
- Critic承認済みのレポートが会話履歴にある、またはこれ以上作業不要なら FINISH
"""


def supervisor_node(state: State) -> State:
    messages = [("system", SUPERVISOR_PROMPT)] + state["messages"]
    decision = llm.with_structured_output(RouteDecision).invoke(messages)
    return {"next": decision.next}


search_tool = TavilySearchResults(max_results=3)
researcher_agent = create_react_agent(
    llm, tools=[search_tool],
    prompt="あなたはリサーチ専門エージェントです。検索ツールで事実情報を集め、箇条書きで報告してください。",
)


def researcher_node(state: State) -> State:
    result = researcher_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]
    return {"messages": [("ai", f"[Researcher]\n{last.content}")]}


def writer_node(state: State) -> State:
    prompt = [
        ("system", "あなたはレポート執筆の専門エージェントです。会話履歴のリサーチ結果や、"
                   "Criticからの差し戻しフィードバックがあればそれを踏まえてレポートを作成・修正してください。"),
        *state["messages"],
    ]
    response = llm.invoke(prompt)
    return {"messages": [("ai", f"[Writer]\n{response.content}")]}


def critic_node(state: State) -> State:
    revision_count = state.get("revision_count", 0)

    # 上限に達していたら問答無用で承認扱いにする(無限ループ防止)
    if revision_count >= MAX_REVISIONS:
        return {
            "messages": [("ai", "[Critic] 修正上限回数に達したため、現状のレポートを承認します。")],
            "next": "supervisor",
        }

    verdict_llm = llm.with_structured_output(CriticVerdict)
    prompt = [
        ("system", "あなたはレポート品質を審査するCriticです。直前のWriterのレポートについて、"
                   "事実の裏付け・構成・簡潔さを評価してください。問題があればapproved=falseとし、"
                   "具体的な改善指示をfeedbackに書いてください。"),
        *state["messages"],
    ]
    verdict = verdict_llm.invoke(prompt)

    if verdict.approved:
        return {
            "messages": [("ai", "[Critic] 承認しました。")],
            "next": "supervisor",
        }
    else:
        return {
            "messages": [("ai", f"[Critic] 差し戻し: {verdict.feedback}")],
            "next": "Writer",
            "revision_count": revision_count + 1,
        }


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)
graph_builder.add_node("Critic", critic_node)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"Researcher": "Researcher", "Writer": "Writer", "FINISH": END},
)
graph_builder.add_edge("Researcher", "supervisor")
graph_builder.add_edge("Writer", "Critic")
graph_builder.add_conditional_edges(
    "Critic",
    lambda state: state["next"],
    {"Writer": "Writer", "supervisor": "supervisor"},
)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread-critic"}, "recursion_limit": 40}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについてレポートを作って")],
            "revision_count": 0,
        },
        config,
    )

    print("=== 最終出力(承認済みレポート) ===")
    print(result["messages"][-1].content)
