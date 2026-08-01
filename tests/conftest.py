"""テストスイートの隔離: 開発マシンの FUGU_* 環境変数がテスト挙動を変えないよう
収集時に除去する(各テストが必要なら monkeypatch.setenv で明示的に設定する)。"""
import os

for _var in ("FUGU_REQUIRE_APPROVAL", "FUGU_SANDBOX_BACKEND",
             "FUGU_SANDBOX_MEMORY_MB", "FUGU_APPROVAL_TIMEOUT"):
    os.environ.pop(_var, None)
