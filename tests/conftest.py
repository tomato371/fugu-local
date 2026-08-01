# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""テストスイートの隔離: 開発マシンの FUGU_* 環境変数がテスト挙動を変えないよう
収集時に除去する(各テストが必要なら monkeypatch.setenv で明示的に設定する)。"""
import os

for _var in ("FUGU_REQUIRE_APPROVAL", "FUGU_SANDBOX_BACKEND",
             "FUGU_SANDBOX_MEMORY_MB", "FUGU_APPROVAL_TIMEOUT",
             "FUGU_SEARCH", "FUGU_SEARCH_MULTI", "FUGU_SEARCH_BUDGET",
             "FUGU_SEARCH_PRIOR", "FUGU_SEARCH_PARALLEL",
             "FUGU_SEARCH_THRESHOLD", "FUGU_MAV", "FUGU_MAV_N"):
    os.environ.pop(_var, None)
