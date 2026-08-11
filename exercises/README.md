# 課題(Exercises)

TUTORIAL.mdの「自分で調べて拡張してみる課題」を、実際に手を動かせる形にしたものです。
各課題は「未完成のPyファイル(TODO付き)」と「`answers/`フォルダにある解答」がセットになっています。

## 進め方

1. 対象のファイルを開き、`--- TODO ... ---`で囲まれた箇所を自分で埋める
   (必要に応じてTUTORIAL.mdや、課題ファイル冒頭に書かれた参考ドキュメントを確認する)
2. `python exercises/<ファイル名>.py` で実行して動作を確認する
3. 詰まったら、または完成したら `exercises/answers/<同じファイル名>.py` と見比べる
   (解答ファイルの末尾に解説コメントも書いてあります)

`.env`(ANTHROPIC_API_KEY / TAVILY_API_KEY)はプロジェクトルートのものがそのまま使えます。
`python-dotenv`がカレントディレクトリを基準に`.env`を探すため、実行は必ず
**プロジェクトルート(`langgraph-research-agent/`)から**行ってください。

```bash
cd /Users/kameyama/dev/ai-agent/langgraph-research-agent
python exercises/step1_ex1_calculator_tool.py
```

## 課題一覧

| # | 対応Step | ファイル | 内容 |
|---|---|---|---|
| 1-1 | Step1 | `step1_ex1_calculator_tool.py` | 検索ツールに加えて計算ツールを追加し、複数ツールの使い分けを体験する |
| 1-2 | Step1 | `step1_ex2_tool_choice.py` | `tool_choice`(auto/any/none)でツール使用の強制・禁止を比較する |
| 2-1 | Step2 | `step2_ex1_three_way_supervisor.py` | FactCheckerエージェントを追加し、Supervisorの分岐を3方向にする |
| 3-1 | Step3 | `step3_ex1_multi_thread.py` | 異なる`thread_id`で会話が分離されることを`get_state`で確認する |
| 4-1 | Step4 | `step4_ex1_persistent_revision_count.py` | reducerの無いStateフィールドが呼び出しをまたいでどう扱われるか調べる |
| 5-1 | Step5 | `step5_ex1_double_confirmation.py` | `interrupt()`を1ノード1回のルールを守りつつ二段階確認を追加する |

## 難易度の目安

- 1-1, 1-2, 3-1: 難易度低め。LangGraphの基本APIの動きを確認する課題。
- 2-1, 4-1: 難易度中。既存コードの構造を理解した上での拡張。
- 5-1: 難易度やや高め。ノード分割の設計判断が必要(なぜ1ノードで2回interruptしないのか、
  という理由まで理解できると良い)。

## 補足: 課題ファイルが参照している元コード

`step3_ex1_multi_thread.py`と`step4_ex1_persistent_revision_count.py`は、
プロジェクトルートの`step3_memory.py`/`step4_critic_loop.py`をそのまま`import`して再利用しています。
これは「グラフをモジュールとして定義しておけば、他のスクリプトや将来的にはAPIサーバーからも
使い回せる」というLangGraphアプリの一般的な構成方法の一例でもあります。
