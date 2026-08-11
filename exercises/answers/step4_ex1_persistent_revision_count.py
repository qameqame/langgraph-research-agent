"""
解答: 課題4-1 revision_countの永続化挙動
======================================
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.messages import HumanMessage
from step4_critic_loop import graph


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "ex4-thread"}, "recursion_limit": 40}

    result1 = graph.invoke(
        {
            "messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについて短いレポートを作って")],
            "revision_count": 0,
        },
        config,
    )
    state_after_1 = graph.get_state(config)
    print(f"1回目終了時の revision_count: {state_after_1.values.get('revision_count')}")

    # パターンX: revision_countを明示的に0にリセットして2回目を呼ぶ
    result2_x = graph.invoke(
        {
            "messages": [HumanMessage(content="もっと短く要約して")],
            "revision_count": 0,  # ← Checkpointerの値を上書きしてしまう
        },
        config,
    )
    state_after_2x = graph.get_state(config)
    print(f"パターンX後の revision_count: {state_after_2x.values.get('revision_count')}")

    # ここで一度スレッドをリセットし、パターンYをフェアに比較できるようにする
    config_y = {"configurable": {"thread_id": "ex4-thread-y"}, "recursion_limit": 40}
    graph.invoke(
        {
            "messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについて短いレポートを作って")],
            "revision_count": 0,
        },
        config_y,
    )

    # パターンY: revision_countキー自体を渡さずに2回目を呼ぶ
    result2_y = graph.invoke(
        {
            "messages": [HumanMessage(content="もっと短く要約して")],
            # revision_count キーを渡さない
        },
        config_y,
    )
    state_after_2y = graph.get_state(config_y)
    print(f"パターンY後の revision_count: {state_after_2y.values.get('revision_count')}")

# 解説:
# - LangGraphのStateのうち、Annotated[..., reducer]で集約関数を指定していない
#   フィールド(revision_countはただの int)は、ノードが返した値でそのまま
#   「上書き」されるのがデフォルト挙動。
# - graph.invoke()に渡す初期入力(input)自体も、実は「特殊な最初のノード」が
#   状態を上書きする形で扱われる。そのため、パターンXのように毎回
#   "revision_count": 0 を渡すと、Checkpointerに保存されていた値があっても
#   0にリセットされてしまう。
# - パターンYのようにキー自体を渡さなければ、渡さなかったフィールドは
#   Checkpointerに保存されていた前回の値がそのまま維持される。
# - 実務上の教訓: 「一度セットしたら基本は増減だけしたい」フィールド
#   (カウンタ、フラグなど)は、2回目以降のinvoke呼び出しで不用意に
#   初期値を渡さないよう注意する。あるいはoperator.addのようなreducerを
#   カスタムで指定し、「渡された値を加算する」という挙動に変えることもできる:
#
#     from typing import Annotated
#     import operator
#     revision_count: Annotated[int, operator.add]
#
#   この場合、ノードが返す値は「差分」として扱われるようになる点に注意
#   (詳しくはPersistenceドキュメントのreducerの節を参照)。
