"""fugu_prompts — プロンプト設定レイヤ (Doc D Phase 5)。

fugu_local のプロンプト定数(モジュールグローバルの str)を、コードを書き換えずに
差し替えるための層。``overrides/<NAME>.txt`` が存在するときだけその内容が使われ、
**override が1つも無ければ挙動は完全に不変**(既定プロンプトはコード内のまま)。

prompt_evolver (fugu_evolve) が進化の勝者をここへ書き込み、C3 Workspace 経由で
``auto-evolve/prompts-*`` ブランチにコミットする。保存先は env
``FUGU_PROMPTS_DIR`` で差し替え可能(テスト・実験用)。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def _override_dir() -> Path:
    return Path(os.environ.get("FUGU_PROMPTS_DIR")
                or Path(__file__).parent / "overrides")


def get_prompt(name: str, default: str) -> str:
    """override があればその内容、無ければ default(読めない場合も default)。"""
    try:
        path = _override_dir() / f"{name}.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if content.strip():
                return content
    except OSError:
        pass
    return default


def set_override(name: str, content: str) -> str:
    """override を書き込む(進化の勝者採用時)。戻り値は書き込んだパス。"""
    directory = _override_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.txt"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:  # py3.9 互換
        fh.write(content)
    return str(path)


def clear_override(name: str) -> bool:
    """override を削除する(既定プロンプトへ戻す)。存在しなければ False。"""
    path = _override_dir() / f"{name}.txt"
    if path.exists():
        path.unlink()
        return True
    return False


def list_overrides() -> Dict[str, str]:
    """現在有効な override 一覧 {name: content}。"""
    directory = _override_dir()
    if not directory.is_dir():
        return {}
    out: Dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        try:
            out[path.stem] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return out


def apply_overrides(namespace: dict) -> int:
    """namespace(通常 fugu_local の globals())の str 定数を override で上書きする。

    override 名と一致する **既存の str キーだけ**を差し替える(未知のキーは
    無視 — タイポした override が新しいグローバルを作らない)。戻り値は適用数。
    """
    applied = 0
    for name, content in list_overrides().items():
        if name in namespace and isinstance(namespace[name], str) and content.strip():
            namespace[name] = content
            applied += 1
    return applied
