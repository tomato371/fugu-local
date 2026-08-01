# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_evolve — 自律自己改善パイプライン (Doc C / sefl-learning.md)。

profiler(健全性計測)→ planner(改善提案)→ workspace(隔離ブランチ)→
evaluator(サンドボックス検証)→ CLI オーケストレーター(Critic 承認+merge)。

全モジュールは Chat / Sandbox / GitClient を引数注入で受け、オフラインテスト
可能(LLM・ネットワーク・実 git 不要のフェイクで全経路を検証できる)。
"""
