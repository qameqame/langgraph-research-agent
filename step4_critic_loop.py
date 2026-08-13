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
- Critic承認後は Writer -> Critic -> supervisor -> END と遷移するが、
  supervisorノードはmessagesに何も追記しない。そのため
  「会話履歴の最後のメッセージ」はCriticの短い承認コメントになってしまい、
  レポート本文にはならない。これを避けるため、Writerが実行されるたびに
  最新のレポート本文をState専用フィールド`final_report`に保存しておき、
  最終出力にはそちらを使う。
- supervisorの「次に何をするか」はLLMの判断に委ねているため、プロンプトの
  意図に反してCritic承認前に"FINISH"を選んでしまうことがある(LLMベースの
  ルーティングは強制力のない"お願い"でしかない)。これを防ぐため、
  `report_approved`フラグをCriticが承認した時だけTrueにし、supervisorから
  ENDへ向かう直前に「本当に承認済みか」を機械的にチェックする安全装置
  (`route_from_supervisor`)を設けている。
"""

import os
import sys
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
from pydantic import BaseModel

load_dotenv()

# .envが読み込めていない/キーが未設定だと、API呼び出しがエラーにならず
# 延々とリトライして「固まったように見える」ことがあるため、起動時にチェックする
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("警告: ANTHROPIC_API_KEYが設定されていません(.envを確認してください)", file=sys.stderr)
if not os.environ.get("TAVILY_API_KEY"):
    print("警告: TAVILY_API_KEYが設定されていません(.envを確認してください)", file=sys.stderr)

MEMBERS = ["Researcher", "Writer"]
MAX_REVISIONS = 2  # Critic差し戻しの上限回数

llm = ChatOllama(model="qwen3:30b", temperature=0, client_kwargs={"timeout": 60})

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
    final_report: str  # Writerが書いた最新のレポート本文(Critic承認後の表示用)
    report_approved: bool  # Criticが実際に承認したかどうか(supervisorの誤判定を防ぐガード用)


class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str  # 差し戻す場合の具体的な改善指示


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}

会話履歴の各メッセージ本文の先頭にある [Researcher] / [Writer] / [Critic] という
タグを手がかりに、次の基準で「上から順に」機械的に判断してください。
曖昧な場合や「もう十分そうだ」と感じても、自己判断でFINISHを選ばないでください。

判断基準(上から順に確認すること):
1. 会話履歴に [Researcher] から始まるメッセージが1件も無い場合 → Researcher
2. [Researcher] のメッセージはあるが、[Writer] から始まるメッセージが無い場合 → Writer
3. [Writer] のメッセージはあるが、会話履歴に
   「[Critic] 承認しました。」または
   「[Critic] 修正上限回数に達したため、現状のレポートを承認します。」
   のいずれかのメッセージが見当たらない場合 → Writer
   (Writerの直後には必ずCriticが自動実行されます。Criticの承認メッセージが
   会話履歴で確認できるまでは、絶対にFINISHを選ばないでください)
4. 上記1〜3のいずれにも当てはまらない場合(=Criticの承認が確認できた場合)のみ → FINISH
"""


def supervisor_node(state: State) -> State:
    print("[supervisor] 判断中...", flush=True)
    messages = [("system", SUPERVISOR_PROMPT)] + state["messages"]
    decision = llm.with_structured_output(RouteDecision).invoke(messages)
    print(f"[supervisor] next={decision.next}", flush=True)
    return {"next": decision.next}


search_tool = TavilySearchResults(max_results=3)
researcher_agent = create_react_agent(
    llm, tools=[search_tool],
    prompt=TODAY_NOTE + "\n\n"
           "あなたはリサーチ専門エージェントです。検索ツールで事実情報を集め、箇条書きで報告してください。",
)


def researcher_node(state: State) -> State:
    print("[Researcher] 検索中...(内部でLLM呼び出し+Tavily検索を複数回行うため時間がかかります)", flush=True)
    result = researcher_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]
    print("[Researcher] 完了", flush=True)
    return {"messages": [("ai", f"[Researcher]\n{last.content}")]}


MIN_REPORT_CHARS = 200  # これより短い応答は「全文ではなく差分/コメント」とみなして再試行する
WRITER_MAX_ATTEMPTS = 3


def writer_node(state: State) -> State:
    print("[Writer] 執筆中...", flush=True)
    base_prompt = [
        ("system", TODAY_NOTE + "\n\n"
                   "あなたはレポート執筆の専門エージェントです。会話履歴のリサーチ結果や、"
                   "Criticからの差し戻しフィードバックがあればそれを踏まえてレポートを作成・修正してください。\n"
                   "重要: 差し戻しへの対応であっても、変更点や差分だけを返すのではなく、"
                   "タイトル・本文・フッターを含むレポート全文を毎回最初から最後まで"
                   "省略せずに出力してください。"),
        *state["messages"],
    ]

    response = llm.invoke(base_prompt)
    attempts = 1
    # プロンプトで「全文を出力して」と指示しても、LLMが差分やコメントだけを
    # 返してしまうことがある(指示は"お願い"でしかなく強制力が無いため)。
    # 文字数で機械的に検証し、短すぎれば指示を強めて再試行する。
    while len(response.content) < MIN_REPORT_CHARS and attempts < WRITER_MAX_ATTEMPTS:
        print(f"[Writer] 応答が{len(response.content)}文字と短すぎるため再試行します"
              f"({attempts}/{WRITER_MAX_ATTEMPTS})", flush=True)
        # 注意: ここに("system", ...)を追加すると、Anthropic APIが
        # 「system messageは先頭にまとめて置く必要がある」というルールに反し、
        # "Received multiple non-consecutive system messages" エラーになる。
        # そのため、やり直し指示は("human", ...)としてConversationに積む。
        retry_prompt = base_prompt + [
            ("ai", response.content),
            ("human", f"直前の応答は{len(response.content)}文字しかなく、レポートとして短すぎます。"
                      "差分やコメントではなく、タイトル・本文・フッターを含む完全なレポート全文を"
                      f"{MIN_REPORT_CHARS}文字以上で出力し直してください。"),
        ]
        print("[Writer] 再試行のAPI呼び出し中...", flush=True)
        response = llm.invoke(retry_prompt)
        attempts += 1

    print("[Writer] 完了", flush=True)

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
        "final_report": response.content,  # Critic差し戻しがあれば次のWriter実行で上書きされる
    }


def critic_node(state: State) -> State:
    revision_count = state.get("revision_count", 0)
    print(f"[Critic] 審査中...(revision_count={revision_count})", flush=True)

    # 上限に達していたら問答無用で承認扱いにする(無限ループ防止)
    if revision_count >= MAX_REVISIONS:
        print("[Critic] 修正上限到達のため強制承認", flush=True)
        return {
            "messages": [("ai", "[Critic] 修正上限回数に達したため、現状のレポートを承認します。")],
            "next": "supervisor",
            "report_approved": True,
        }

    verdict_llm = llm.with_structured_output(CriticVerdict)
    prompt = [
        ("system", "あなたはレポート品質を審査するCriticです。直前のWriterのレポートについて、"
                   "事実の裏付け・構成・簡潔さを評価してください。問題があればapproved=falseとし、"
                   "具体的な改善指示をfeedbackに書いてください。"),
        *state["messages"],
    ]
    verdict = verdict_llm.invoke(prompt)
    print(f"[Critic] approved={verdict.approved}", flush=True)

    if verdict.approved:
        return {
            "messages": [("ai", "[Critic] 承認しました。")],
            "next": "supervisor",
            "report_approved": True,
        }
    else:
        return {
            "messages": [("ai", f"[Critic] 差し戻し: {verdict.feedback}")],
            "next": "Writer",
            "revision_count": revision_count + 1,
            "report_approved": False,
        }


def route_from_supervisor(state: State) -> str:
    """supervisorの判断(state["next"])を実際の遷移先に変換する。

    LLMの判断は絶対ではないため、"FINISH"を選んでいても
    Criticがまだ承認していなければ機械的に差し戻す安全装置を入れている。

    差し戻し先は「まだ何も進んでいない(=最初のユーザーメッセージしか無い)なら
    Researcherへ、それ以外(Researcherが既に走っているなど何かしら進捗がある)なら
    Writerへ」という基準にしている。final_reportの有無で判定すると、
    「Researcherは終わったがWriterがまだ一度も走っていない」ケースで
    Researcherへ差し戻され続けてしまう(実際に今回ハマったのはこのケース)。
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
    print(result.get("final_report", "(レポートが生成されませんでした。revision_countやログを確認してください)"))
