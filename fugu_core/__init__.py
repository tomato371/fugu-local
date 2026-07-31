"""fugu_core — MoA オーケストレーターのミドルウェア層 (Doc D / phase-4.md)。

memory(エピソード記憶)/ compressor(状態圧縮)/ pipeline(投機実行)/
debate(スコア行列と討論)。全モジュールは env フラグ + lazy import で
fugu_local にフックされ、フラグ未設定時の既定経路は完全に不変。依存は全て
注入(Chat/Embedder/Sandbox)でオフラインテスト可能。
"""
