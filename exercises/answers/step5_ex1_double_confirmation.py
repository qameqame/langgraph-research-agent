"""
解答: 課題5-1 二段階確認(final_confirmation)の追加
======================================
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
- 人間の最終確認まで完了したレポートがあるなら FINISH
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
    return {"messages": [("ai", f"[Researcher]\n{result['messages'][-1].content}")]}


def writer_node(state: State) -> State:
    prompt = [
        ("system", "あなたはレポート執筆の専門エージェントです。会話履歴のリサーチ結果や、"
                   "差し戻しフィードバックがあればそれを踏まえて修正してください。"),
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

    verdict = llm.with_structured_output(CriticVerdict).invoke([
        ("system", "あなたはレポート品質を審査するCriticです。問題があればapproved=falseとし、"
                   "具体的な改善指示をfeedbackに書いてください。"),
        *state["messages"],
    ])

    if verdict.approved:
        return {"messages": [("ai", "[Critic] 自動レビュー承認。人間の確認へ回します。")], "next": "human_approval"}
    else:
        return {
            "messages": [("ai", f"[Critic] 差し戻し: {verdict.feedback}")],
            "next": "Writer",
            "revision_count": revision_count + 1,
        }


def human_approval_node(state: State) -> State:
    last_report = state["messages"][-1].content
    decision = interrupt({"question": "このレポートを承認しますか?", "report": last_report})
    if decision.get("approved"):
        return {"messages": [("ai", "[Human] 承認しました。最終確認へ進みます。")], "next": "final_confirmation"}
    else:
        feedback = decision.get("feedback", "")
        return {
            "messages": [("ai", f"[Human] 差し戻し: {feedback}")],
            "next": "Writer",
            "revision_count": state.get("revision_count", 0) + 1,
        }


def final_confirmation_node(state: State) -> State:
    last_report = state["messages"][-1].content
    decision = interrupt({
        "question": "本当にこのレポートで確定してよいですか?(この操作は取り消せません)",
        "report": last_report,
    })
    if decision.get("approved"):
        return {"messages": [("ai", "[Human] 最終確定しました。")], "next": "supervisor"}
    else:
        feedback = decision.get("feedback", "")
        return {
            "messages": [("ai", f"[Human] 最終確認で差し戻し: {feedback}")],
            "next": "Writer",
            "revision_count": state.get("revision_count", 0) + 1,
        }


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)
graph_builder.add_node("Critic", critic_node)
graph_builder.add_node("human_approval", human_approval_node)
graph_builder.add_node("final_confirmation", final_confirmation_node)

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
    {"Writer": "Writer", "final_confirmation": "final_confirmation"},
)
graph_builder.add_conditional_edges(
    "final_confirmation",
    lambda state: state["next"],
    {"Writer": "Writer", "supervisor": "supervisor"},
)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "ex5-thread"}, "recursion_limit": 40}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについてレポートを作って")],
            "revision_count": 0,
        },
        config,
    )

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n=== {payload['question']} ===")
        print(payload["report"])
        answer = input("\ny/n: ").strip().lower()
        if answer == "y":
            resume_value = {"approved": True}
        else:
            resume_value = {"approved": False, "feedback": input("差し戻し理由: ")}
        result = graph.invoke(Command(resume=resume_value), config)

    print("\n=== 完成レポート ===")
    print(result["messages"][-1].content)

# 解説:
# - interrupt()は「1ノードにつき1回」を守り、確認ステップを増やしたい場合は
#   ノード自体を分けるのが安全。human_approvalとfinal_confirmationは
#   ほぼ同じ形のノードだが、質問文と次の遷移先(next)が異なるだけの
#   独立したノードとして定義している。
# - human_approval承認後のnextを"supervisor"から"final_confirmation"に
#   変更し、条件付きEdgeの行き先マップにも"final_confirmation"を追加する
#   必要がある(ノードを足しただけでは自動的に繋がらない)。
# - mainのwhileループ側は変更不要な点に注目。__interrupt__の有無だけを
#   見て汎用的にy/nを聞く作りにしていたため、確認ステップが増えても
#   呼び出し側のロジックは共通のまま使い回せる。
