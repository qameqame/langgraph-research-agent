"""
課題3-1: 異なるthread_idで会話が分離されることを確認する
======================================
Step3で学んだCheckpointerは`thread_id`ごとに状態を分けて保存します。
この課題では、既存の`step3_memory.py`のグラフをそのまま再利用し、
2つの異なるthread_id("thread-A", "thread-B")で別々の話題の会話をした後、
それぞれの状態が混ざっていないことを`graph.get_state()`で確認します。

参考ドキュメント:
- Persistence(スレッドとcheckpointの関係): https://docs.langchain.com/oss/python/langgraph/persistence
- get_state の使い方: https://reference.langchain.com/python/langgraph

進め方:
1. 下のTODOを埋める
2. `python exercises/step3_ex1_multi_thread.py` を実行
3. 2つのスレッドで会話の中身(メッセージ数・内容)が異なることを確認する
4. `exercises/answers/step3_ex1_multi_thread.py` と見比べる
"""

import os
import sys

# 親フォルダ(langgraph-research-agent直下)のstep3_memory.pyを再利用する
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage
from step3_memory import graph  # Step3で作ったグラフをそのままimport


if __name__ == "__main__":
    # --- TODO 1: 2つの異なるthread_idのconfigを用意する -----------------
    config_a = None  # {"configurable": {"thread_id": "thread-A"}, "recursion_limit": 25}
    config_b = None  # {"configurable": {"thread_id": "thread-B"}, "recursion_limit": 25}
    # --- TODO 1 ここまで ---------------------------------------------------

    if config_a is None or config_b is None:
        print("TODO 1 を埋めてから実行してください")
        raise SystemExit

    # --- TODO 2: thread-Aで1つ目の話題について会話する ---------------------
    # result_a = graph.invoke(
    #     {"messages": [HumanMessage(content="生成AIエージェントの2026年ビジネス活用トレンドについてレポートを作って")]},
    #     config_a,
    # )
    # --- TODO 2 ここまで -----------------------------------------------------

    # --- TODO 3: thread-Bで全く別の話題について会話する ------------------------
    # result_b = graph.invoke(
    #     {"messages": [HumanMessage(content="日本のスタートアップ資金調達市場の動向についてレポートを作って")]},
    #     config_b,
    # )
    # --- TODO 3 ここまで -----------------------------------------------------

    # --- TODO 4: それぞれのスレッドの状態を取得し、分離を確認する ------------------
    # state_a = graph.get_state(config_a)
    # state_b = graph.get_state(config_b)
    # print(f"thread-A のメッセージ数: {len(state_a.values['messages'])}")
    # print(f"thread-B のメッセージ数: {len(state_b.values['messages'])}")
    # print(f"thread-A 最初のユーザー発言: {state_a.values['messages'][0].content}")
    # print(f"thread-B 最初のユーザー発言: {state_b.values['messages'][0].content}")
    # --- TODO 4 ここまで -----------------------------------------------------
