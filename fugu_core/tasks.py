# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_core.tasks — 永続タスクボード (Doc E Phase 2)。

conduct() の単発フラットプランでは扱えない「複数サブタスクの段階的消化」と
「プロセスを跨いだ再開」を提供する。完走性(長時間・多段階タスクを無人で
走り切る)が目的 — 賢さではなく、途中で死んでも続きから再開できることが本体。

- :class:`TodoItem` / :class:`TaskBoard` — サブタスク列+依存関係(blocked_by)。
  ボードは1ファイル=1JSONで毎更新スナップショット永続化(壊れたファイルは
  load が None を返すだけで、実行系を止めない)。
- :func:`decompose` — スキーマ制約 LLM 分解。depends_on は「自分より前の項目
  のみ」を許す(構造的に循環しない)。失敗時は単一タスクへの決定論的
  フォールバック(planner._fallback_proposals と同じ思想)。

フックは ``FUGU_TASKS=1`` のときだけ fugu_answer の入口が分解を試み、
2件以上に分解できた場合のみボード経由の消化ループへ移行する(単一タスク・
失敗時は従来経路のままでオーバーヘッドゼロ)。CLI ``--resume <board_id>``
で未完了ボードを続きから消化できる。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

STATUSES = ("pending", "in_progress", "completed", "failed")

#: サブタスク数の上限(1件の質問を無限に割らない)。
MAX_ITEMS = 5

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["subject"],
            },
        }
    },
    "required": ["subtasks"],
}

_SYSTEM = (
    "You are a task planner for a local answering pipeline. Split the user's "
    "request into a SMALL ordered list of concrete subtasks only when the "
    "request genuinely needs multiple steps; return a single subtask (or the "
    "request itself) when it does not. depends_on lists 1-based indices of "
    "EARLIER subtasks whose results are required. Reply with JSON only."
)


@dataclass
class TodoItem:
    id: str
    subject: str
    status: str = "pending"
    blocked_by: List[str] = field(default_factory=list)
    result: str = ""   # 完了時の成果(後続タスクへのコンテキスト)
    attempts: int = 0  # リトライで消費した回数(チェックポイントに永続化)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:24]
    return slug or "board"


def _boards_dir() -> str:
    return (os.environ.get("FUGU_TASKS_DIR")
            or os.path.join(os.path.expanduser("~"), ".fugu_tasks"))


class TaskBoard:
    """サブタスク列の永続ボード。毎更新でスナップショット保存する。"""

    def __init__(self, board_id: str, question: str, items: List[TodoItem],
                 directory: Optional[str] = None):
        self.board_id = board_id
        self.question = question
        self.items = items
        self.directory = directory or _boards_dir()

    # ---------- 生成・入出力 ----------

    @classmethod
    def new(cls, question: str, items: List[TodoItem],
            directory: Optional[str] = None,
            now_fn: Optional[Callable[[], str]] = None) -> "TaskBoard":
        now = now_fn or (lambda: time.strftime("%Y%m%d-%H%M%S"))
        board = cls(f"{_slugify(question)}-{now()}", question, items,
                    directory=directory)
        board.save()
        return board

    @classmethod
    def load(cls, board_id: str,
             directory: Optional[str] = None) -> Optional["TaskBoard"]:
        """ボードを読み込む。無い・壊れている場合は None(実行系を止めない)。"""
        directory = directory or _boards_dir()
        path = os.path.join(directory, f"{board_id}.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            items = [TodoItem(
                id=str(raw["id"]), subject=str(raw["subject"]),
                status=str(raw.get("status", "pending")),
                blocked_by=[str(x) for x in raw.get("blocked_by", [])],
                result=str(raw.get("result", "")),
                attempts=int(raw.get("attempts", 0) or 0),
            ) for raw in obj["items"]]
            return cls(str(obj["board_id"]), str(obj.get("question", "")),
                       items, directory=directory)
        except Exception:
            return None

    def save(self) -> str:
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"{self.board_id}.json")
        payload = {"board_id": self.board_id, "question": self.question,
                   "items": [asdict(item) for item in self.items]}
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        return path

    # ---------- 状態遷移 ----------

    def get(self, item_id: str) -> Optional[TodoItem]:
        return next((item for item in self.items if item.id == item_id), None)

    def update(self, item_id: str, status: str, result: str = "") -> None:
        item = self.get(item_id)
        if item is None or status not in STATUSES:
            return
        item.status = status
        if result:
            item.result = result
        self.save()  # チェックポイント: 毎遷移で永続化(途中死しても再開可能)

    def retry(self, item_id: str, max_retries: int) -> bool:
        """失敗した項目をリトライ枠内なら pending に戻す(完走性: 一過性の失敗で
        依存チェーンを永久停止させない)。枠超過・未知 id は False。"""
        item = self.get(item_id)
        if item is None or item.attempts >= max(0, max_retries):
            return False
        item.attempts += 1
        item.status = "pending"
        item.result = ""
        self.save()
        return True

    def reset_stale(self) -> int:
        """クラッシュ・強制終了で in_progress のまま残った項目を pending に戻す。

        実測 2026-08-01: 実行中プロセスを kill すると当該サブタスクが
        in_progress で永続化され、next_ready()(pending のみ対象)が二度と
        拾えず、依存する後続も永久に ready にならなかった。再開の入口で必ず
        呼ぶこと。戻り値はリセットした件数。
        """
        stale = [item for item in self.items if item.status == "in_progress"]
        for item in stale:
            item.status = "pending"
        if stale:
            self.save()
        return len(stale)

    def next_ready(self) -> Optional[TodoItem]:
        """依存が全て completed の pending を(定義順で)1件返す。無ければ None。

        依存タスクが failed のものは永遠に ready にならない — 消化ループは
        そこで自然に止まり、統合時に未実行として報告される。
        """
        done = {item.id for item in self.items if item.status == "completed"}
        for item in self.items:
            if item.status == "pending" and all(b in done for b in item.blocked_by):
                return item
        return None

    def all_done(self) -> bool:
        return all(item.status in ("completed", "failed") for item in self.items)

    def progress(self) -> str:
        done = sum(1 for item in self.items if item.status == "completed")
        return f"{done}/{len(self.items)} completed"

    def results_context(self, chars_per: int = 500) -> str:
        """後続サブタスクへ注入する「先行結果ダイジェスト」("" if none)。"""
        done = [item for item in self.items
                if item.status == "completed" and item.result]
        if not done:
            return ""
        lines = ["## 先行サブタスクの結果 (task board)"]
        for item in done:
            lines.append(f"### {item.subject}")
            lines.append(item.result[:chars_per])
        return "\n".join(lines)


def decompose(question: str, chat, max_items: int = MAX_ITEMS) -> List[TodoItem]:
    """質問をサブタスク列に分解する(失敗・単純な質問は1件のフォールバック)。

    depends_on は 1-based の「自分より前の項目」のみ有効(前方参照・自己参照は
    黙って落とす — インデックス構造上、循環は作れない)。
    """
    prompt = (
        f"Request: {question}\n\n"
        f'Respond with {{"subtasks": [{{"subject": ..., "depends_on": [...]}}]}} '
        f"({max_items} subtasks at most)."
    )
    try:
        raw = chat.complete(prompt, system=_SYSTEM, fmt=DECOMPOSE_SCHEMA,
                            temperature=0.0)
        obj = json.loads(raw)
        entries = obj.get("subtasks", []) if isinstance(obj, dict) else []
        items: List[TodoItem] = []
        for idx, entry in enumerate(entries[:max_items], start=1):
            if not isinstance(entry, dict):
                continue
            subject = entry.get("subject")
            if not (isinstance(subject, str) and subject.strip()):
                continue
            deps_raw = entry.get("depends_on")
            deps = [f"t{d}" for d in deps_raw
                    if isinstance(d, int) and 1 <= d < idx] \
                if isinstance(deps_raw, list) else []
            items.append(TodoItem(id=f"t{idx}", subject=subject.strip(),
                                  blocked_by=deps))
        if items:
            return items
    except Exception:
        pass
    return [TodoItem(id="t1", subject=question)]


def synthesize_board(board: TaskBoard) -> str:
    """消化済みボードを最終回答テキストへ決定論的に統合する。

    LLM を使わない(統合の失敗で完走が壊れないこと最優先)。完了サブタスクの
    結果を見出し付きで並べ、失敗・未実行は明示的に報告する(silent-drop しない)。
    """
    completed = [item for item in board.items
                 if item.status == "completed" and item.result]
    parts: List[str] = []
    for item in completed:
        parts.append(f"## {item.subject}\n{item.result}")
    leftovers = [item for item in board.items if item.status != "completed"]
    if leftovers:
        lines = ["## 未完了サブタスク"]
        for item in leftovers:
            note = f" — {item.result}" if item.result else ""
            lines.append(f"- [{item.status}] {item.subject}{note}")
        parts.append("\n".join(lines))
    if not parts:
        return "(タスクボードに結果がありません)"
    return "\n\n".join(parts)
