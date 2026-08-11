"""
Step 5(最終): human-in-the-loop 承認フロー
======================================
Step4のグラフに「人間の最終承認」を追加し、これでハンズオンの完成形とする。

構成:
    ... -> Writer -> Critic -> (承認: human_approval / 差し戻し: Writer)
    human_approval -> (人間がOK: supervisor(→FINISH) / 人間がNG: Writer)

ポイント:
- interrupt() を呼ぶとグラフの実行がそこで一時停止し、呼び出し元(main)に
  制御が戻る。checkpointerがあるため、途中状態は保存されている。
- 人間の入力(承認 or 差し戻し理由)を Command(resume=...) で渡すと、
  interrupt() の返り値としてノードの処理が再開される。
- Critic(自動レビュー)を通過したものだけを人間がチェックすればよいので、
  レビューコストを最小限にしつつ品質を担保できる。

これでStep1〜5を積み上げた「リサーチ&レポート作成マルチエージェント」が完成。
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
from langgraph.types import interrupt, Command
from pydantic import BaseModel

load_dotenv()

MEMBERS = ["Researcher", "Writer"]
MAX_REVISIONS = 2

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    revision_count: int


class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}
判断基準:
- まだ十分な情報が集まっていない場合は Researcher
- 情報は揃っていてレポートがまだ無い場合は Writer
- 人間承認済みのレポートが会話履歴にあるなら FINISH
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
                   "Critic・人間からの差し戻しフィードバックがあればそれを踏まえて修正してください。"),
        *state["messages"],
    ]
    response = llm.invoke(prompt)
    return {"messages": [("ai", f"[Writer]\n{response.content}")]}


def critic_node(state: State) -> State:
    revision_count = state.get("revision_count", 0)

    if revision_count >= MAX_REVISIONS:
        return {
            "messages": [("ai", "[Critic] 修正上限回数に達したため、人間の判断に委ねます。")],
            "next": "human_approval",
        }

    verdict_llm = llm.with_structured_output(CriticVerdict)
    prompt = [
        ("system", "あなたはレポート品質を審査するCriticです。事実の裏付け・構成・簡潔さを評価し、"
                   "問題があればapproved=falseとして具体的な改善指示をfeedbackに書いてください。"),
        *state["messages"],
    ]
    verdict = verdict_llm.invoke(prompt)

    if verdict.approved:
        return {
            "messages": [("ai", "[Critic] 自動レビュー承認。人間の最終確認へ回します。")],
            "next": "human_approval",
        }
    else:
        return {
            "messages": [("ai", f"[Critic] 差し戻し: {verdict.feedback}")],
            "next": "Writer",
            "revision_count": revision_count + 1,
        }


def human_approval_node(state: State) -> State:
    last_report = state["messages"][-1].content
    decision = interrupt(
        {
            "question": "このレポートを承認しますか?",
            "report": last_report,
        }
    )
    # decision は main側から Command(resume=...) で渡された値
    if decision.get("approved"):
        return {
            "messages": [("ai", "[Human] 承認しました。")],
            "next": "supervisor",
        }
    else:
        feedback = decision.get("feedback", "")
        return {
            "messages": [("ai", f"[Human] 差し戻し: {feedback}")],
            "next": "Writer",
            "revision_count": state.get("revision_count", 0) + 1,
        }


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)
graph_builder.add_node("Critic", critic_node)
graph_builder.add_node("human_approval", human_approval_node)

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
    {"Writer": "Writer", "human_approval": "human_approval"},
)
graph_builder.add_conditional_edges(
    "human_approval",
    lambda state: state["next"],
    {"Writer": "Writer", "supervisor": "supervisor"},
)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread-hitl"}, "recursion_limit": 40}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについてレポートを作って")],
            "revision_count": 0,
        },
        config,
    )

    # interrupt()が呼ばれるとグラフはここで停止し、__interrupt__キーが結果に含まれる
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== 人間の承認待ち ===")
        print(payload["report"])
        answer = input("\nこのレポートを承認しますか? (y/n): ").strip().lower()

        if answer == "y":
            resume_value = {"approved": True}
        else:
            feedback = input("差し戻し理由を入力してください: ")
            resume_value = {"approved": False, "feedback": feedback}

        result = graph.invoke(Command(resume=resume_value), config)

    print("\n=== 完成レポート(人間承認済み) ===")
    print(result["messages"][-1].content)
