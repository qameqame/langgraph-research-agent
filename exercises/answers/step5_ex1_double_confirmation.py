"""
解答: 課題5-1 二段階確認(final_confirmation)の追加
======================================
"""

from datetime import date
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
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

llm = ChatOllama(model="qwen3:8b", temperature=0, model_kwargs={"think": False})

# with_structured_output(JSON構造化出力)はthink無効化と相性が悪く、
# 無効なJSONを返すことがある(Ollama/Qwen3の既知の問題)。
# 構造化出力が必要な呼び出し(Supervisorのルーティング判断、Criticの審査)には
# thinkingを有効なままにした専用のLLMインスタンスを使う。
router_llm = ChatOllama(model="qwen3:8b", temperature=0)

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
    revision_count: int
    final_report: str  # Writerが書いた最新のレポート本文(承認後の表示用)
    report_approved: bool  # 最終確認まで承認されたかどうか(supervisorの誤判定を防ぐガード用)


class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}

会話履歴の各メッセージ本文の先頭にある [Researcher] / [Writer] / [Critic] / [Human] という
タグを手がかりに、次の基準で「上から順に」機械的に判断してください。
曖昧な場合や「もう十分そうだ」と感じても、自己判断でFINISHを選ばないでください。

判断基準(上から順に確認すること):
1. 会話履歴に [Researcher] から始まるメッセージが1件も無い場合 → Researcher
2. [Researcher] のメッセージはあるが、[Writer] から始まるメッセージが無い場合 → Writer
3. [Writer] のメッセージはあるが、Criticの承認メッセージが見当たらない場合 → Writer
4. Criticの承認メッセージはあるが、「[Human] 最終確定しました。」というメッセージが
   見当たらない場合 → Writer
   (一次承認だけでなく、最終確認まで完了したことが確認できるまでは
   絶対にFINISHを選ばないでください)
5. 上記1〜4のいずれにも当てはまらない場合(=最終確定が確認できた場合)のみ → FINISH
"""


def _rule_based_route(state: State) -> str:
    """LLMの構造化出力(JSON)が失敗したときの、決定的な代替ルーティング。
    SUPERVISOR_PROMPTに書いてある判断基準と同じロジックをコードでも再現している。"""
    contents = [str(getattr(m, "content", "")) for m in state["messages"]]
    if not any(c.startswith("[Researcher]") for c in contents):
        return "Researcher"
    if not any(c.startswith("[Writer]") for c in contents):
        return "Writer"
    critic_approved = any(
        c.startswith("[Critic] 自動レビュー承認。人間の確認へ回します。")
        or c.startswith("[Critic] 修正上限回数に達したため、人間の判断に委ねます。")
        for c in contents
    )
    if not critic_approved:
        return "Writer"
    finalized = any(c.startswith("[Human] 最終確定しました。") for c in contents)
    if not finalized:
        return "Writer"
    return "FINISH"


def supervisor_node(state: State) -> State:
    messages = [("system", SUPERVISOR_PROMPT)] + state["messages"]
    try:
        decision = router_llm.with_structured_output(RouteDecision).invoke(messages)
        return {"next": decision.next}
    except Exception as e:
        # qwen3:8bのような小さいモデルは、会話履歴が長くなると構造化出力(JSON)を
        # 正しく生成できず、レポート本文の続きを書こうとしてしまうことがある
        # (Ollama/Qwen3のjson_schema構造化出力まわりの既知の不安定さ)。
        # LLMが失敗した場合はクラッシュさせず、会話履歴から機械的に次のノードを
        # 決めるルールベースの代替ルーティングにフォールバックする。
        print(f"[supervisor] 構造化出力の解析に失敗しました: {e}", flush=True)
        fallback = _rule_based_route(state)
        print(f"[supervisor] ルールベースのフォールバックで next={fallback} を選択します", flush=True)
        return {"next": fallback}


search_tool = TavilySearchResults(max_results=3)
researcher_agent = create_react_agent(
    llm, tools=[search_tool],
    prompt=TODAY_NOTE + "\n\n"
           "あなたはリサーチ専門エージェントです。検索ツールで事実情報を集め、箇条書きで報告してください。",
)


def researcher_node(state: State) -> State:
    result = researcher_agent.invoke({"messages": state["messages"]})
    return {"messages": [("ai", f"[Researcher]\n{result['messages'][-1].content}")]}


MIN_REPORT_CHARS = 200  # これより短い応答は「全文ではなく差分/コメント」とみなして再試行する
WRITER_MAX_ATTEMPTS = 3


def writer_node(state: State) -> State:
    base_prompt = [
        ("system", TODAY_NOTE + "\n\n"
                   "あなたはレポート執筆の専門エージェントです。会話履歴のリサーチ結果や、"
                   "差し戻しフィードバックがあればそれを踏まえて修正してください。\n"
                   "重要: 差し戻しへの対応であっても、変更点や差分だけを返すのではなく、"
                   "タイトル・本文・フッターを含むレポート全文を毎回最初から最後まで"
                   "省略せずに出力してください。"),
        *state["messages"],
    ]

    response = llm.invoke(base_prompt)
    attempts = 1
    while len(response.content) < MIN_REPORT_CHARS and attempts < WRITER_MAX_ATTEMPTS:
        print(f"[Writer] 応答が{len(response.content)}文字と短すぎるため再試行します"
              f"({attempts}/{WRITER_MAX_ATTEMPTS})", flush=True)
        # 注意: ここに("system", ...)を追加すると"Received multiple
        # non-consecutive system messages"エラーになるため、("human", ...)で渡す。
        retry_prompt = base_prompt + [
            ("ai", response.content),
            ("human", f"直前の応答は{len(response.content)}文字しかなく、レポートとして短すぎます。"
                      "差分やコメントではなく、タイトル・本文・フッターを含む完全なレポート全文を"
                      f"{MIN_REPORT_CHARS}文字以上で出力し直してください。"),
        ]
        response = llm.invoke(retry_prompt)
        attempts += 1

    if len(response.content) < MIN_REPORT_CHARS:
        print(f"[Writer] 警告: {attempts}回試しても{MIN_REPORT_CHARS}文字以上の応答が"
              "得られませんでした。前回のレポートを維持します。", flush=True)
        fallback_report = state.get("final_report") or response.content
        return {
            "messages": [("ai", f"[Writer] (警告: 短い応答のため前回のレポートを維持)\n{response.content}")],
            "final_report": fallback_report,
        }

    return {
        "messages": [("ai", f"[Writer]\n{response.content}")],
        "final_report": response.content,
    }


def critic_node(state: State) -> State:
    revision_count = state.get("revision_count", 0)
    if revision_count >= MAX_REVISIONS:
        return {
            "messages": [("ai", "[Critic] 修正上限回数に達したため、人間の判断に委ねます。")],
            "next": "human_approval",
        }

    try:
        verdict = router_llm.with_structured_output(CriticVerdict).invoke([
            ("system", "あなたはレポート品質を審査するCriticです。問題があればapproved=falseとし、"
                       "具体的な改善指示をfeedbackに書いてください。"),
            *state["messages"],
        ])
    except Exception as e:
        # supervisor_node同様、構造化出力(JSON)の解析に失敗することがある。
        # ここで例外を投げるとグラフ全体がクラッシュしてしまうため、
        # 安全側(承認しない=Writerに差し戻す)にフォールバックする。
        # revision_countの上限ガードがあるので、無限ループにはならない。
        print(f"[Critic] 構造化出力の解析に失敗しました: {e}", flush=True)
        print("[Critic] 安全側のフォールバックとして差し戻し扱いにします", flush=True)
        verdict = CriticVerdict(
            approved=False,
            feedback="(自動審査の構造化出力に失敗したため、内容を再確認して出力し直してください)",
        )

    if verdict.approved:
        return {"messages": [("ai", "[Critic] 自動レビュー承認。人間の確認へ回します。")], "next": "human_approval"}
    else:
        return {
            "messages": [("ai", f"[Critic] 差し戻し: {verdict.feedback}")],
            "next": "Writer",
            "revision_count": revision_count + 1,
        }


def human_approval_node(state: State) -> State:
    # state["messages"][-1] は直前のCriticノードが追記した短いコメントであり、
    # レポート本文ではない。表示には必ずfinal_reportを使うこと。
    last_report = state.get("final_report", "(レポートが見つかりませんでした)")
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
    # ここでも同様に、messages[-1]は直前の[Human]承認コメントであり
    # レポート本文ではないため、final_reportを使う。
    last_report = state.get("final_report", "(レポートが見つかりませんでした)")
    decision = interrupt({
        "question": "本当にこのレポートで確定してよいですか?(この操作は取り消せません)",
        "report": last_report,
    })
    if decision.get("approved"):
        return {
            "messages": [("ai", "[Human] 最終確定しました。")],
            "next": "supervisor",
            "report_approved": True,
        }
    else:
        feedback = decision.get("feedback", "")
        return {
            "messages": [("ai", f"[Human] 最終確認で差し戻し: {feedback}")],
            "next": "Writer",
            "revision_count": state.get("revision_count", 0) + 1,
            "report_approved": False,
        }


def route_from_supervisor(state: State) -> str:
    """supervisorの判断(state["next"])を実際の遷移先に変換する。

    LLMの判断は絶対ではないため、"FINISH"を選んでいても
    最終確認までの承認が済んでいなければ機械的に差し戻す安全装置を入れている。

    差し戻し先は「まだ何も進んでいない(=最初のユーザーメッセージしか無い)なら
    Researcherへ、それ以外ならWriterへ」という基準にしている。final_reportの
    有無で判定すると、「Researcherは終わったがWriterがまだ一度も走っていない」
    ケースでResearcherへ差し戻され続けてしまう。
    """
    decision = state["next"]
    if decision == "FINISH" and not state.get("report_approved", False):
        if len(state.get("messages", [])) <= 1:
            return "Researcher"
        return "Writer"
    return decision


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
    route_from_supervisor,
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
    print(result.get("final_report", "(レポートが生成されませんでした。revision_countやログを確認してください)"))

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
# - Critic/human_approval/final_confirmationのいずれもmessagesには短い
#   コメントしか追記しないため、result["messages"][-1]をそのまま表示すると
#   レポート本文ではなく承認コメントが出てしまう(Step4/Step5本編と同じ罠)。
#   Writer実行時にfinal_reportへ本文を保存しておくことで回避している。
