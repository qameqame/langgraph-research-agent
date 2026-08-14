"""
Step 3: メモリ永続化(会話を継続する)
======================================
Step2までは graph.invoke() を呼ぶたびに状態がリセットされていた。
ここでは Checkpointer を追加し、thread_id ごとに状態を保存・復元できるようにする。

これにより:
    「レポート作って」 -> 「もっと短くして」
のような複数ターンのやり取りが、直前の文脈を保ったまま可能になる。

本番運用ではSqliteSaver/PostgresSaverなどの永続ストレージに差し替え可能。
ここでは学習用にインメモリのMemorySaverを使う。
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
from pydantic import BaseModel

load_dotenv()

MEMBERS = ["Researcher", "Writer"]
llm = ChatOllama(model="qwen3:8b", temperature=0, model_kwargs={"think": False})

# with_structured_output(JSON構造化出力)はthink無効化と相性が悪く、
# 無効なJSONを返すことがある(Ollama/Qwen3の既知の問題)。
# 構造化出力が必要な呼び出し(Supervisorのルーティング判断)には
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


class RouteDecision(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}
判断基準:
- まだ十分な情報が集まっていない場合は Researcher
- 情報は揃っていてレポートがまだ無い/修正が必要な場合は Writer
- レポートが完成し、これ以上作業が不要な場合は FINISH
※ユーザーからの追加依頼(短くして、等)がある場合もWriterに再依頼すること。
"""


def _rule_based_route(state: State) -> str:
    """LLMの構造化出力(JSON)が失敗したときの、決定的な代替ルーティング。
    会話履歴に何が積まれているかを機械的にチェックする
    (SUPERVISOR_PROMPTに書いてある判断基準と同じロジックをコードでも再現している)。
    注意: これは簡易版のため、「一度FINISHした後にターン2で修正依頼が来た」
    ケースは正しく扱えない(Writerが既にいるとFINISH扱いになってしまう)。
    あくまでLLMが失敗した際の最終防御であり、通常はLLMの判断が優先される。"""
    contents = [str(getattr(m, "content", "")) for m in state["messages"]]
    if not any(c.startswith("[Researcher]") for c in contents):
        return "Researcher"
    if not any(c.startswith("[Writer]") for c in contents):
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
    last = result["messages"][-1]
    return {"messages": [("ai", f"[Researcher]\n{last.content}")]}


def writer_node(state: State) -> State:
    prompt = [
        ("system", TODAY_NOTE + "\n\n"
                   "あなたはレポート執筆の専門エージェントです。会話履歴(過去のリサーチ結果や、"
                   "ユーザーからの修正依頼)を踏まえてレポートを作成・修正してください。"),
        *state["messages"],
    ]
    response = llm.invoke(prompt)
    return {"messages": [("ai", f"[Writer]\n{response.content}")]}


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

# --- ここがStep2との差分: checkpointerを渡してcompile ---
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":
    # thread_idが同じなら会話が継続される(別スレッドなら完全に別文脈になる)
    config = {"configurable": {"thread_id": "demo-thread-1"}, "recursion_limit": 25}

    print("=== ターン1: レポート作成依頼 ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについてレポートを作って")]},
        config,
    )
    print(result["messages"][-1].content)

    print("\n=== ターン2: 修正依頼(前の文脈を引き継ぐ) ===")
    result = graph.invoke(
        {"messages": [HumanMessage(content="3行程度に要約して")]},
        config,
    )
    print(result["messages"][-1].content)
