# IDE 統合エンドポイント (fugu_api)

VS Code / Cursor 拡張などエディタ統合向けの低レイテンシ API。フル MoA パイプライン
(`POST /ask`) と異なり、いずれも **単一モデル呼び出し**(または純サンドボックス実行)。

起動: `uvicorn fugu_api:app --host 0.0.0.0 --port 8000`

## POST /completion — インライン補完

カーソル位置のコード続きを生成する。コーダーモデル (`qwen3-coder:30b`) 1回呼び出し。

```json
// request
{"prefix": "def fib(n):\n    ", "suffix": "", "language": "python", "max_tokens": 256}
// response
{"completion": "if n < 2:\n        return n\n    ...", "elapsed_seconds": 1.9}
```

```bash
curl -s localhost:8000/completion -H 'Content-Type: application/json' \
  -d '{"prefix": "def fib(n):\n    "}'
```

- `max_tokens` は Ollama の `num_predict` にそのまま渡る (1–2048)。
- モデル失敗は HTTP 502、Ollama 未起動は HTTP 503。

## POST /refactor — 指示付きリライト + diff

```json
// request
{"code": "for i in range(10):\n    print(i)", "instruction": "use a while loop", "language": "python"}
// response
{"refactored": "i = 0\nwhile i < 10:\n    print(i)\n    i += 1", "diff": "--- before\n+++ after\n...", "elapsed_seconds": 3.0}
```

`diff` は `difflib.unified_diff`(fromfile=before, tofile=after)。エディタ側で
そのままパッチプレビューに使える。

## POST /test-run — サンドボックス実行 / TDC / 自己デバッグ

3モード(排他、上から優先):

1. **`tests` 指定** — `solution.py` + `test_solution.py` を一時ディレクトリに並べ
   pytest を実行 (fugu_tdc.run_tests)。LLM 不要。
2. **`max_retries > 0`** — 実行失敗時に stderr をコーダーモデルへ渡して修正→再実行
   (fugu_sandbox.run_with_self_debug)。要 Ollama。
3. **どちらも無し** — 1回だけサンドボックス実行。LLM 不要。

```json
// request
{"code": "print('hi')", "max_retries": 0, "timeout": 30}
// response
{"ok": true, "stdout": "hi\n", "stderr": "", "exit_code": 0,
 "timed_out": false, "attempts": 1, "code": "print('hi')"}
```

```bash
# TDC モード: テストを green にできるか検証
curl -s localhost:8000/test-run -H 'Content-Type: application/json' -d '{
  "code": "def add(a, b):\n    return a + b",
  "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3"
}'
```

- 実行は常に一時ディレクトリ・stdin=DEVNULL・timeout kill (fugu_sandbox)。
- `code` フィールドは自己デバッグ後の最終コード(修正されなかった場合は入力のまま)。
