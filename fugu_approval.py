"""fugu_approval — 人間承認ゲート (Doc E Phase 3)。

自律実行が長時間・無人で回るほど「危険操作の直前で人間に確認する」口が必要に
なる。本モジュールは ``FUGU_REQUIRE_APPROVAL=1`` のときだけ有効になるブロッキング
承認ゲートを提供する(未設定なら常に即 True — 既定経路は完全に不変):

- :func:`require_approval` — 承認要求を登録して待つ。fugu_local の _emit 経由で
  ``approval_required`` イベントが SSE/WS に流れ、コンソールにも run_id が出る。
  タイムアウト・拒否は **False(安全側=実行しない)**。
- :func:`resolve` — fugu_api の ``POST /approve/{run_id}`` から承認/拒否を確定。
- :func:`pending` — 未決の run_id 一覧(``GET /approvals``)。

ゲート箇所(いずれも FUGU_REQUIRE_APPROVAL=1 のときのみ):
fugu_sandbox の任意コード実行(``run()``)と fugu_evolve の merge 直前。
pytest 等の固定 argv 実行(``run_argv``)は対象外 — 検証ループが承認連打に
ならないよう「任意コードの入口」だけを守る。
"""
from __future__ import annotations

import os
import threading
import uuid
from typing import Dict, List, Optional

#: 承認待ちの既定タイムアウト(秒)。超過は拒否扱い。
DEFAULT_TIMEOUT = 300.0

_LOCK = threading.Lock()
_PENDING: Dict[str, threading.Event] = {}
_DECISIONS: Dict[str, bool] = {}


def _emit_event(event: str, **data) -> None:
    """fugu_local のイベント基盤へ転送(未 import・失敗は無視)。"""
    try:
        import fugu_local
        fugu_local._emit(event, **data)
    except Exception:
        pass


def require_approval(kind: str, detail: str,
                     timeout: Optional[float] = None) -> bool:
    """危険操作の直前で承認を待つ。

    ``FUGU_REQUIRE_APPROVAL`` が "1" 以外なら即 True(ゲート無効)。有効時は
    run_id を発行して ``approval_required`` イベントを発火し、:func:`resolve`
    が呼ばれるまで(最大 timeout 秒)ブロックする。タイムアウト・拒否は False。
    """
    if os.environ.get("FUGU_REQUIRE_APPROVAL") != "1":
        return True
    if timeout is None:
        timeout = float(os.environ.get("FUGU_APPROVAL_TIMEOUT") or DEFAULT_TIMEOUT)
    run_id = f"{kind}-{uuid.uuid4().hex[:8]}"
    event = threading.Event()
    with _LOCK:
        _PENDING[run_id] = event
    _emit_event("approval_required", run_id=run_id, kind=kind,
                detail=detail[:200])
    print(f"   [approval] 承認待ち: {run_id} ({kind}) — "
          f"POST /approve/{run_id} で承認/拒否")
    granted = event.wait(timeout)
    with _LOCK:
        decision = _DECISIONS.pop(run_id, False)
        _PENDING.pop(run_id, None)
    approved = bool(granted and decision)
    if not approved:
        print(f"   [approval] {'タイムアウト' if not granted else '拒否'}: "
              f"{run_id} → 実行しません")
    return approved


def resolve(run_id: str, approve: bool) -> bool:
    """承認/拒否を確定する。該当 run_id が無ければ False(既に解決済み含む)。"""
    with _LOCK:
        event = _PENDING.get(run_id)
        if event is None:
            return False
        _DECISIONS[run_id] = bool(approve)
    event.set()
    return True


def pending() -> List[str]:
    """未決の承認要求 run_id 一覧。"""
    with _LOCK:
        return list(_PENDING)
