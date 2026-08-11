"""
課題5-1: 二段階確認(final_confirmation)を追加する
======================================
Step5では`human_approval`ノードで1回だけ`interrupt()`を呼んでいました。
この課題では、承認後にもう一段階「本当に確定してよいですか?(取り消せません)」という
最終確認を追加します。

重要な設計上の注意:
  1つのノードの中で`interrupt()`を2回連続で呼ぶのは推奨されません
  (公式ドキュメントでも、決定的でない繰り返し呼び出しは避けるよう明記されています)。
  そのため、確認ステップは**別ノードに分離**し、条件付きEdgeでつなぐのが正しいやり方です。

構成イメージ:
    ... Critic(承認) --> human_approval --> final_confirmation --> supervisor
                              │                    │
                          (差し戻し)Writer      (差し戻し)Writer

参考ドキュメント:
- Interrupts(複数回のinterrupt呼び出しに関する注意点):
  https://docs.langchain.com/oss/python/langgraph/interrupts
- interrupt リファレンス: https://reference.langchain.com/python/langgraph/types/interrupt

進め方:
1. 下のTODOを埋める(final_confirmation_nodeの実装とグラフへの組み込み)
2. `python exercises/step5_ex1_double_confirmation.py` を実行し、
   承認 → 最終確認、の2段階の入力が求められることを確認する
3. `exercises/answers/step5_ex1_double_confirmation.py` と見比べる
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
        return {"messages": [("ai", "[Human] 承認しました。")], "next": "final_confirmation"}
    else:
        feedback = decision.get("feedback", "")
        return {
            "messages": [("ai", f"[Human] 差し戻し: {feedback}")],
            "next": "Writer",
            "revision_count": state.get("revision_count", 0) + 1,
        }


# --- TODO 1: final_confirmation_node を実装する ---------------------------
# ヒント: human_approval_node と同じ形。ただし今度は「本当に確定してよいですか?
#         (この操作は取り消せません)」という質問文でinterrupt()を呼ぶ。
#         承認されたら next="supervisor"、差し戻されたら next="Writer" とし、
#         revision_countも忘れずインクリメントする。
#
# def final_confirmation_node(state: State) -> State:
#     ...

# --- TODO 1 ここまで --------------------------------------------------------


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("Researcher", researcher_node)
graph_builder.add_node("Writer", writer_node)
graph_builder.add_node("Critic", critic_node)
graph_builder.add_node("human_approval", human_approval_node)
# --- TODO 2: final_confirmationノードをグラフに追加する ---------------------
# graph_builder.add_node("final_confirmation", final_confirmation_node)
# --- TODO 2 ここまで --------------------------------------------------------

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
# --- TODO 3: human_approvalの行き先に final_confirmation を追加する ----------
graph_builder.add_conditional_edges(
    "human_approval",
    lambda state: state["next"],
    {"Writer": "Writer", "final_confirmation": "final_confirmation"},  # ← "final_confirmation": "final_confirmation" が正しく動くにはTODO1/2が必要
)
# --- TODO 4: final_confirmationからの分岐(承認->supervisor / 差し戻し->Writer)を追加する ---
# graph_builder.add_conditional_edges(
#     "final_confirmation",
#     lambda state: state["next"],
#     {"Writer": "Writer", "supervisor": "supervisor"},
# )
# --- TODO 3/4 ここまで -------------------------------------------------------

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
