"""
解答: 課題3-1 マルチスレッドの分離確認
======================================
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.messages import HumanMessage
from step3_memory import graph


if __name__ == "__main__":
    config_a = {"configurable": {"thread_id": "thread-A"}, "recursion_limit": 25}
    config_b = {"configurable": {"thread_id": "thread-B"}, "recursion_limit": 25}

    result_a = graph.invoke(
        {"messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについてレポートを作って")]},
        config_a,
    )

    result_b = graph.invoke(
        {"messages": [HumanMessage(content="日本のスタートアップ資金調達市場の動向についてレポートを作って")]},
        config_b,
    )

    state_a = graph.get_state(config_a)
    state_b = graph.get_state(config_b)

    print(f"thread-A のメッセージ数: {len(state_a.values['messages'])}")
    print(f"thread-B のメッセージ数: {len(state_b.values['messages'])}")
    print(f"thread-A 最初のユーザー発言: {state_a.values['messages'][0].content}")
    print(f"thread-B 最初のユーザー発言: {state_b.values['messages'][0].content}")

# 解説:
# - MemorySaverはプロセス内のメモリ上に {thread_id: [checkpoint, ...]} という
#   形で状態を保持している。thread_idが異なれば参照するキーも異なるため、
#   会話内容が混ざることはない。
# - graph.get_state(config) はStateSnapshotというオブジェクトを返し、
#   .values でグラフのStateの中身(messages, next など)にアクセスできる。
# - もし同じthread_idを使ってしまうと、2つ目の話題の会話が1つ目の会話履歴に
#   追記され、Supervisorやエージェントが古い話題の文脈を引きずってしまう。
#   会話ID(ユーザーIDやセッションIDなど)からthread_idを一意に決めることが
#   実運用では重要になる。
