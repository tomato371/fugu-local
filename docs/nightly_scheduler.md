# 無人実行の配線 — nightly 自己改善とタスクボード再開

`fugu_evolve --nightly` はマーカー(evolution_history に "mode: nightly" と記録)で
あり、スケジューラ自体は OS 側に配線する。Windows は Task Scheduler を使う。

## nightly 自己改善(毎日 03:00、1提案・PR モード)

merge まで無人にしたくない場合は `--pr-mode` を付け、翌朝 `auto-evolve/*`
ブランチを人間がレビューして merge する運用を推奨する。

```powershell
schtasks /create /tn "fugu-evolve-nightly" /sc daily /st 03:00 `
  /tr "cmd /c cd /d D:\repos\fugu-local-integ && python -m fugu_evolve --repo . --nightly --pr-mode --max-proposals 1 >> %USERPROFILE%\fugu_evolve_nightly.log 2>&1"
```

- 削除: `schtasks /delete /tn "fugu-evolve-nightly" /f`
- 即時試run: `schtasks /run /tn "fugu-evolve-nightly"`
- 完全無人(merge まで自動)にする場合は `--pr-mode` を外す。安全ゲートは
  「pytest 100% + bench 非退行 + LLM Critic 承認 + auto-evolve/ ブランチ隔離」。

## 中断したタスクボードの再開 (FUGU_TASKS)

`FUGU_TASKS=1` での実行中に落ちた場合、ボードは
`%USERPROFILE%\.fugu_tasks\<board_id>.json`(`FUGU_TASKS_DIR` で変更可)に
チェックポイントされている。未完了サブタスクから再開するには:

```powershell
python fugu_local.py --resume <board_id>
```

board_id は実行時ログの `[tasks] ... (board=...)` に表示される。
