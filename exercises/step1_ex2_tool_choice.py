"""
課題1-2: tool_choiceでツール使用を強制してみる
======================================
Step1の`bind_tools(tools)`はデフォルトで「LLMが自分でツールを使うか判断する」
(tool_choice="auto"相当)動作でした。この課題では`tool_choice`を明示的に指定し、
挙動がどう変わるかを実験を通じて確認します。

試すこと:
  A) tool_choice未指定(auto) : 計算不要な質問には検索ツールを使わない
  B) tool_choiceで検索ツールを強制: 計算だけで済む質問でも検索ツールが呼ばれる
  C) tool_choice="none"      : ツールを一切使わず、必ずテキストのみで回答する

参考ドキュメント:
- ChatAnthropic.bind_tools の tool_choice 引数:
  https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic
- Anthropic公式のtool_choice仕様(force_tool_useなど)は上記リファレンスのリンク先や
  Anthropic APIドキュメントも合わせて確認してみてください。

注意(Ollama利用時): ChatOllamaのtool_choiceパラメータは現状Ollama側で
サポートされておらず、渡しても無視されます。そのためこの課題のB/Cのような
「ツール使用を強制/禁止する」挙動はOllamaでは再現できず、A/B/Cどれも
LLMの自主判断(auto相当)と同じ結果になる可能性が高いです。
tool_choiceの制御を確認したい場合は、この課題だけAnthropicのAPIキーに
戻して試すことをおすすめします。

進め方:
1. 下のTODOを埋める
2. 3パターンの挙動を見比べる
3. `exercises/answers/step1_ex2_tool_choice.py` と結果を照らし合わせる
"""

import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

llm = ChatOllama(model="qwen3:30b", temperature=0)

question = "1 + 1 は何ですか?"  # 検索が不要な単純な質問

# --- TODO A: tool_choiceを指定しない(デフォルト=auto)場合のモデルを作る ---
llm_auto = None  # ここを llm.bind_tools(tools) に置き換える
# --- TODO A ここまで -----------------------------------------------------

# --- TODO B: 検索ツールの使用を強制するモデルを作る ------------------------
# ヒント: tool_choice に "any"(どれか1つを必ず使う)、
#         または特定のツール名を渡す方法を調べて試してみましょう。
llm_forced = None
# --- TODO B ここまで -----------------------------------------------------

# --- TODO C: ツールを一切使わせないモデルを作る ---------------------------
# 注意: tool_choice="none"という文字列は「'none'という名前のツールを使え」と
# 解釈されてしまい、存在しないツールとしてAPIエラーになる。
# ツール使用を無効化したい場合は辞書形式で{"type": "none"}を渡す必要がある。
llm_none = None
# --- TODO C ここまで -----------------------------------------------------


def describe(label, response):
    tool_calls = getattr(response, "tool_calls", [])
    print(f"[{label}] tool_calls={[tc['name'] for tc in tool_calls] or 'なし'}")
    # ツールを呼んだ場合、Anthropicのcontentはテキストではなく
    # tool_useブロックを含むリストになり、そのまま表示すると読みにくい。
    # tool_callsが無い(=テキスト回答のみ)場合だけcontentを表示する。
    if tool_calls:
        print(f"[{label}] content=(ツール呼び出しのみでテキスト応答なし)")
    else:
        print(f"[{label}] content={response.content}")


if __name__ == "__main__":
    if llm_auto is None or llm_forced is None or llm_none is None:
        print("TODO A/B/C を埋めてから実行してください")
    else:
        describe("A: auto", llm_auto.invoke(question))
        describe("B: forced", llm_forced.invoke(question))
        describe("C: none", llm_none.invoke(question))
