# 無人実行の配線 — nightly 自己改善とタスクボード再開

`fugu_evolve --nightly` はマーカー(evolution_history に "mode: nightly" と記録)で
あり、スケジューラ自体は OS 側に配線する。Windows は Task Scheduler を使う。

## nightly 自己改善(毎日 03:00、1提案・PR モード)

merge まで無人にしたくない場合は `--pr-mode` を付け、翌朝 `auto-evolve/*`
ブランチを人間がレビューして merge する運用を推奨する。

> **2026-10-10 時点の状態: このタスク `fugu-evolve-nightly` は未登録**（`schtasks /query /fo LIST` に
> 存在しない。`\Nakayama\` 配下にあるのは koyomi・quorum・soroban の 3 つ）。
> 登録する場合の作業ディレクトリは `D:\repos\fugu-local-integ`（現存しない）ではなく、
> 現在のリポジトリ `D:\株式会社中山龍也\R&D\quorum\fugu-local-integ` を指す。

> **登録コマンドは載せない（2026-10-10）**: 現在のパスは `&`（`R&D`）を含むため、
> `schtasks /tr "cmd /c cd /d ..."` の形では cmd の引用が壊れる（同じ事情で、会社フォルダ直下の
> `quorum_start_hidden.vbs` は `&` を含まない場所に置いたラッパーにしてある）。PowerShell の文字列では
> `\"` がエスケープにならないので、`\"` で包んだ書き方（2026-10-10 に一度書いたが、構文エラーになるため取り下げた）も使えない。
> `&` を含まない場所にラッパー（.bat／.vbs）を置くか、`New-ScheduledTaskAction -WorkingDirectory` で
> 作業ディレクトリを指定して `Register-ScheduledTask` に渡す。**未登録・未検証（2026-10-10）**。
> 旧パス（`D:\repos\...`）時代のコマンドは `git show 5754e0d^:docs/nightly_scheduler.md` にある。

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
