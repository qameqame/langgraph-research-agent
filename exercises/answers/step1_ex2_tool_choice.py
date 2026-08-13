"""
解答: 課題1-2 tool_choiceの実験
======================================

注意(Ollama利用時): ChatOllamaのtool_choiceは現状Ollama側で未サポートのため
無視され、B/Cで意図した強制/禁止の挙動が再現できないことがあります。
tool_choiceの制御を確認したい場合はAnthropicのAPIキーに戻して試してください。
"""

import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

llm = ChatOllama(model="qwen3:30b", temperature=0)

question = "1 + 1 は何ですか?"

# A) auto(デフォルト): LLMが必要性を判断する
llm_auto = llm.bind_tools(tools)

# B) 強制: 必ず何らかのツールを使わせる("any"を指定)
llm_forced = llm.bind_tools(tools, tool_choice="any")

# C) 禁止: ツールを一切使わせない
# 注意: tool_choice="none"という文字列は「'none'という名前のツールを使え」と
# 解釈され、存在しないツールとしてAPIエラーになる(実際にハマったポイント)。
# ツール使用を無効化する場合は辞書形式で{"type": "none"}を渡す。
llm_none = llm.bind_tools(tools, tool_choice={"type": "none"})


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
# 補足: response.contentの形について。ツールを呼ばない場合(A, C)はcontentが
# 単純な文字列になるが、ツールを呼ぶ場合(B)はcontentがテキストブロック/
# tool_useブロックなどを含む「リスト」になる(今回はテキスト部分が空で
# tool_useブロックのみだったため、生のdict列がそのまま見える)。
# ツール呼び出しの有無を確認したいときはcontentではなくtool_callsを見ること。
#
# 参考: ChatAnthropic.bind_toolsのtool_choiceは、"auto" / "any" / 特定のツール名(str)
# は文字列のショートカットとして受け付けるが、"none"だけは特別扱いされておらず
# 「'none'という名前のツール」を指定したとみなされてしまう。ツール使用を無効化
# したい場合は必ず辞書形式 {"type": "none"} を渡すこと。
# こうした文字列ショートカットの対応範囲はバージョンによって変わりうるため、
# 実装前に必ずリファレンスの最新版を確認すること:
# https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic
