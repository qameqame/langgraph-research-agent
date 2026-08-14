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
    next: Literal["Researcher", "Writer", "FactChecker", "FINISH"]


SUPERVISOR_PROMPT = f"""あなたはリサーチ&レポート作成チームの管理者です。
メンバー: {MEMBERS}

判断基準:
- まだ十分な情報が集まっていない場合は Researcher
- 情報は揃っていてレポートがまだ無い場合は Writer
- レポートは書けたが、数値や固有名詞の裏取りがまだの場合は FactChecker
- FactCheckerの検証を通過したレポートがあれば FINISH
"""


def _rule_based_route(state: State) -> str:
    """LLMの構造化出力(JSON)が失敗したときの、決定的な代替ルーティング。
    会話履歴に何が積まれているかを機械的にチェックする
    (SUPERVISOR_PROMPTに書いてある判断基準と同じロジックをコードでも再現している)。"""
    contents = [str(getattr(m, "content", "")) for m in state["messages"]]
    if not any(c.startswith("[Researcher]") for c in contents):
        return "Researcher"
    if not any(c.startswith("[Writer]") for c in contents):
        return "Writer"
    if not any(c.startswith("[FactChecker]") for c in contents):
        return "FactChecker"
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


MIN_REPORT_CHARS = 200  # Writerの応答がこれより短い場合は「全文でない」とみなして再試行する
MIN_FACTCHECK_CHARS = 10  # FactCheckerの応答がこれより短い(=ほぼ空)場合は再試行する
MAX_RETRY_ATTEMPTS = 2


def writer_node(state: State) -> State:
    base_prompt = [
        ("system", TODAY_NOTE + "\n\n"
                   "あなたはレポート執筆の専門エージェントです。会話履歴のリサーチ結果を元に、"
                   "簡潔で読みやすい日本語レポートを作成してください。\n"
                   "重要: タイトル・本文・結論を含むレポート全文を、省略せずに出力してください。"),
        *state["messages"],
    ]
    response = llm.invoke(base_prompt)
    attempts = 1
    # 小さいモデルは、指示通り「全文」を書かず短い応答で終わらせてしまうことがある。
    # 文字数で機械的に検証し、短すぎれば指示を強めて再試行する(Step4と同じパターン)。
    while len(response.content) < MIN_REPORT_CHARS and attempts < MAX_RETRY_ATTEMPTS:
        print(f"[Writer] 応答が{len(response.content)}文字と短すぎるため再試行します"
              f"({attempts}/{MAX_RETRY_ATTEMPTS})", flush=True)
        retry_prompt = base_prompt + [
            ("ai", response.content),
            ("human", f"直前の応答は{len(response.content)}文字しかなく、レポートとして短すぎます。"
                      f"タイトル・本文・結論を含む完全なレポート全文を{MIN_REPORT_CHARS}文字以上で"
                      "出力し直してください。"),
        ]
        response = llm.invoke(retry_prompt)
        attempts += 1

    if len(response.content) < MIN_REPORT_CHARS:
        print(f"[Writer] 警告: {attempts}回試しても{MIN_REPORT_CHARS}文字以上の応答が"
              "得られませんでした。得られた内容をそのまま採用します。", flush=True)

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
    content = result["messages"][-1].content
    attempts = 1
    # 小さいモデルは、長い会話履歴を渡されるとReActループの最後で空/ほぼ空の
    # 応答を返してしまうことがある。空でないことを機械的に確認し、
    # 空だった場合は指示を強めて再試行する。
    while len(content.strip()) < MIN_FACTCHECK_CHARS and attempts < MAX_RETRY_ATTEMPTS:
        print(f"[FactChecker] 応答が空/短すぎる({len(content.strip())}文字)ため再試行します"
              f"({attempts}/{MAX_RETRY_ATTEMPTS})", flush=True)
        retry_messages = list(state["messages"]) + [
            ("human", "直前の検証結果が空でした。検証結果を必ずテキストで出力してください。"
                      "問題が無ければ『検証OK: 主要な主張は裏付けが取れました』とだけ出力してください。"),
        ]
        result = fact_checker_agent.invoke({"messages": retry_messages})
        content = result["messages"][-1].content
        attempts += 1

    if len(content.strip()) < MIN_FACTCHECK_CHARS:
        print(f"[FactChecker] 警告: {attempts}回試しても有効な応答が得られませんでした。"
              "デフォルトのメッセージを使用します。", flush=True)
        content = "(検証結果を取得できませんでした。手動で確認してください)"

    return {"messages": [("ai", f"[FactChecker]\n{content}")]}


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
