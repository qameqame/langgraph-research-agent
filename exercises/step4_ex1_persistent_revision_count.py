"""
課題4-1: revision_countがCheckpointerをまたいでどう扱われるか調べる
======================================
`step4_critic_loop.py`のStateには`revision_count`というreducer無しのフィールドがあります。
reducerが無いフィールドは、ノードが返した値で「上書き」されるのがLangGraphのデフォルト挙動です。

この課題では、同じthread_idに対して2回`graph.invoke()`を呼び出す際、
2回目の呼び出しで`revision_count`をどう扱うかによって結果がどう変わるかを実験します。

  パターンX: 2回目のinvoke時に "revision_count": 0 を明示的に渡す
             → Checkpointerに保存されていた値を上書きしてリセットしてしまう
  パターンY: 2回目のinvoke時に "revision_count" キー自体を渡さない
             → Checkpointerに保存されていた前回の値がそのまま使われる

参考ドキュメント:
- Persistence(状態更新のマージ/上書きルール): https://docs.langchain.com/oss/python/langgraph/persistence
- add_messagesのようなreducerを自作フィールドに付ける方法も調べてみましょう
  (Annotated[int, ...]の形でカスタムreducer関数を指定できます)

進め方:
1. 下のTODOを埋める(2パターンのinvoke呼び出し)
2. 実行してrevision_countの値の違いを確認する
3. `exercises/answers/step4_ex1_persistent_revision_count.py` と見比べる
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage
from step4_critic_loop import graph


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "ex4-thread"}, "recursion_limit": 40}

    # 1回目: 通常通りレポート作成を依頼(revision_countは0から開始)
    result1 = graph.invoke(
        {
            "messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについて短いレポートを作って")],
            "revision_count": 0,
        },
        config,
    )
    state_after_1 = graph.get_state(config)
    print(f"1回目終了時の revision_count: {state_after_1.values.get('revision_count')}")

    # --- TODO: パターンXを試す(revision_countを0にリセットして2回目を呼ぶ) -------
    # result2_x = graph.invoke(
    #     {
    #         "messages": [HumanMessage(content="もっと短く要約して")],
    #         "revision_count": 0,   # ← ここが上書きのポイント
    #     },
    #     config,
    # )
    # state_after_2x = graph.get_state(config)
    # print(f"パターンX後の revision_count: {state_after_2x.values.get('revision_count')}")
    # --- TODO ここまで -------------------------------------------------------------

    # --- TODO: パターンYを試す(revision_countキーを渡さずに2回目を呼ぶ) -----------
    # result2_y = graph.invoke(
    #     {
    #         "messages": [HumanMessage(content="もっと短く要約して")],
    #         # revision_count キーを意図的に渡さない
    #     },
    #     config,
    # )
    # state_after_2y = graph.get_state(config)
    # print(f"パターンY後の revision_count: {state_after_2y.values.get('revision_count')}")
    # --- TODO ここまで -------------------------------------------------------------

    print("\nTODOを埋めて、パターンX/Yでrevision_countがどう変わるか比較してください。")
