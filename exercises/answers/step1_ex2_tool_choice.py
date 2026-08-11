"""
解答: 課題1-2 tool_choiceの実験
======================================
"""

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

question = "1 + 1 は何ですか?"

# A) auto(デフォルト): LLMが必要性を判断する
llm_auto = llm.bind_tools(tools)

# B) 強制: 必ず何らかのツールを使わせる("any"を指定)
llm_forced = llm.bind_tools(tools, tool_choice="any")

# C) 禁止: ツールを一切使わせない
llm_none = llm.bind_tools(tools, tool_choice="none")


def describe(label, response):
    tool_calls = getattr(response, "tool_calls", [])
    print(f"[{label}] tool_calls={[tc['name'] for tc in tool_calls] or 'なし'}")
    print(f"[{label}] content={response.content}")


if __name__ == "__main__":
    describe("A: auto", llm_auto.invoke(question))
    describe("B: forced(any)", llm_forced.invoke(question))
    describe("C: none", llm_none.invoke(question))

# 解説:
# - A(auto): 「1+1」は検索不要なので、tool_callsは空でテキスト回答のみが返る。
# - B(any): tool_choice="any" にすると、LLMは必ずどれかのツールを呼ぶことを
#   強制される。今回は登録ツールが検索だけなので、不要なはずの検索が実行される。
#   これは「本来なら要らない検索コスト・レイテンシが発生する」ことを意味する。
# - C(none): ツール使用が禁止されるため、たとえ必要でもツールは呼ばれず、
#   モデルは自分の知識だけでテキスト回答を返そうとする。
#
# 実務的な学び: tool_choice="auto"(デフォルト)が多くの場合で適切。
# "any"や特定ツール名を強制するのは、「必ずこのツールの結果を経由させたい」
# ような特殊なワークフロー(例: 必ず検索結果に基づいて回答させたいRAG的な用途)
# に限定して使うのが安全。
#
# 参考: ChatAnthropicのtool_choiceは "auto" | "any" | "none" | 特定のツール名(str)
# を受け付ける。正確な受理値は必ずリファレンスで最新版を確認すること:
# https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic
