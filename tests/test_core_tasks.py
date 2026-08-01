"""fugu_core.tasks のオフラインテスト + FUGU_TASKS=1 フック検証。"""
import json

from fugu_llm import FakeChat
from fugu_core.tasks import (
    TaskBoard,
    TodoItem,
    decompose,
    synthesize_board,
)


def _plan_json(*subtasks):
    return json.dumps({"subtasks": list(subtasks)})


# ------------------------------------------------------------------ decompose

def test_decompose_parses_subtasks_and_dependencies():
    chat = FakeChat(responses=[_plan_json(
        {"subject": "調査する", "depends_on": []},
        {"subject": "実装する", "depends_on": [1]},
        {"subject": "検証する", "depends_on": [2]},
    )])
    items = decompose("作って検証して", chat)
    assert [i.id for i in items] == ["t1", "t2", "t3"]
    assert items[0].blocked_by == []
    assert items[1].blocked_by == ["t1"]
    assert items[2].blocked_by == ["t2"]


def test_decompose_drops_forward_and_self_references():
    chat = FakeChat(responses=[_plan_json(
        {"subject": "a", "depends_on": [1, 2, 5]},  # 自己参照・前方参照は無効
        {"subject": "b", "depends_on": [1]},
    )])
    items = decompose("q", chat)
    assert items[0].blocked_by == []       # t1 は誰にも依存できない
    assert items[1].blocked_by == ["t1"]


def test_decompose_skips_invalid_entries():
    chat = FakeChat(responses=[_plan_json(
        {"subject": "  "}, "junk", {"subject": "valid"})])
    items = decompose("q", chat)
    assert [i.subject for i in items] == ["valid"]


def test_decompose_caps_items():
    chat = FakeChat(responses=[_plan_json(
        *[{"subject": f"s{i}"} for i in range(10)])])
    assert len(decompose("q", chat, max_items=3)) == 3


def test_decompose_fallback_on_junk_and_exception():
    items = decompose("my question", FakeChat(default="not json"))
    assert len(items) == 1 and items[0].subject == "my question"
    boom = FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError()))
    assert decompose("my question", boom)[0].id == "t1"


# ------------------------------------------------------------------ TaskBoard

def _board(tmp_path, items=None):
    items = items or [TodoItem(id="t1", subject="one"),
                      TodoItem(id="t2", subject="two", blocked_by=["t1"])]
    return TaskBoard.new("the question", items, directory=str(tmp_path),
                         now_fn=lambda: "20260801-120000")


def test_board_save_load_roundtrip(tmp_path):
    board = _board(tmp_path)
    assert board.board_id == "the-question-20260801-120000"
    loaded = TaskBoard.load(board.board_id, directory=str(tmp_path))
    assert loaded is not None
    assert loaded.question == "the question"
    assert [i.id for i in loaded.items] == ["t1", "t2"]
    assert loaded.items[1].blocked_by == ["t1"]


def test_board_update_checkpoints_to_disk(tmp_path):
    board = _board(tmp_path)
    board.update("t1", "completed", result="done result")
    reloaded = TaskBoard.load(board.board_id, directory=str(tmp_path))
    assert reloaded.items[0].status == "completed"
    assert reloaded.items[0].result == "done result"


def test_board_load_missing_or_corrupt_is_none(tmp_path):
    assert TaskBoard.load("nope", directory=str(tmp_path)) is None
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    assert TaskBoard.load("bad", directory=str(tmp_path)) is None


def test_next_ready_respects_dependencies(tmp_path):
    board = _board(tmp_path)
    assert board.next_ready().id == "t1"   # t2 は t1 待ち
    board.update("t1", "completed")
    assert board.next_ready().id == "t2"
    board.update("t2", "completed")
    assert board.next_ready() is None
    assert board.all_done() is True


def test_failed_dependency_blocks_dependents_forever(tmp_path):
    board = _board(tmp_path)
    board.update("t1", "failed", result="boom")
    assert board.next_ready() is None      # t2 は永遠に ready にならない
    assert board.all_done() is False       # 未消化が残っていることは分かる


def test_progress_and_results_context(tmp_path):
    board = _board(tmp_path)
    assert board.progress() == "0/2 completed"
    board.update("t1", "completed", result="x" * 1000)
    ctx = board.results_context(chars_per=100)
    assert ctx.startswith("## 先行サブタスクの結果")
    assert "### one" in ctx
    assert len(ctx) < 300  # chars_per で切り詰め
    assert TaskBoard.new("q", [TodoItem(id="t1", subject="s")],
                         directory=str(tmp_path)).results_context() == ""


# ------------------------------------------------------------------ synthesize

def test_synthesize_assembles_completed_and_reports_leftovers(tmp_path):
    board = _board(tmp_path)
    board.update("t1", "completed", result="answer one")
    board.update("t2", "failed", result="error: x")
    text = synthesize_board(board)
    assert "## one\nanswer one" in text
    assert "## 未完了サブタスク" in text
    assert "[failed] two — error: x" in text


def test_synthesize_pending_only_board_reports_leftovers(tmp_path):
    board = TaskBoard.new("q", [TodoItem(id="t1", subject="s")],
                          directory=str(tmp_path))
    text = synthesize_board(board)
    assert "## 未完了サブタスク" in text and "[pending] s" in text


def test_synthesize_itemless_board_is_explicit(tmp_path):
    board = TaskBoard.new("q", [], directory=str(tmp_path))
    assert "結果がありません" in synthesize_board(board)


# ------------------------------------------------------------------ fugu_local hook

def _fake_decompose_two(question, chat, max_items=5):
    return [TodoItem(id="t1", subject="step one"),
            TodoItem(id="t2", subject="step two", blocked_by=["t1"])]


def test_tasked_answer_single_item_falls_through(monkeypatch, tmp_path):
    import fugu_local
    from fugu_core import tasks as tasks_mod
    monkeypatch.setenv("FUGU_TASKS_DIR", str(tmp_path))
    monkeypatch.setattr(tasks_mod, "decompose",
                        lambda q, chat, max_items=5: [TodoItem(id="t1", subject=q)])
    assert fugu_local._tasked_answer("simple question") is None


def test_tasked_answer_runs_board_in_dependency_order(monkeypatch, tmp_path):
    import fugu_local
    from fugu_core import tasks as tasks_mod
    monkeypatch.setenv("FUGU_TASKS_DIR", str(tmp_path))
    monkeypatch.setattr(tasks_mod, "decompose", _fake_decompose_two)
    seen = []

    def fake_answer(question, plan=None, history=None):
        seen.append(question)
        return f"answer to [{question.splitlines()[-1]}]"

    monkeypatch.setattr(fugu_local, "fugu_answer", fake_answer)
    out = fugu_local._tasked_answer("big request")
    assert out is not None
    assert seen[0] == "step one"                    # 依存順に消化
    assert "先行サブタスクの結果" in seen[1]         # 先行結果が後続へ注入される
    assert seen[1].endswith("step two")
    assert "## step one" in out and "## step two" in out


def test_tasked_answer_failed_subtask_reported_not_silent(monkeypatch, tmp_path):
    import fugu_local
    from fugu_core import tasks as tasks_mod
    monkeypatch.setenv("FUGU_TASKS_DIR", str(tmp_path))
    monkeypatch.setattr(tasks_mod, "decompose", _fake_decompose_two)
    monkeypatch.setattr(fugu_local, "fugu_answer",
                        lambda q, plan=None, history=None: "__ERROR__: model down")
    out = fugu_local._tasked_answer("big request")
    assert "## 未完了サブタスク" in out
    assert "[failed] step one" in out
    assert "[pending] step two" in out              # 依存先が失敗 → 未実行のまま報告


def test_run_board_resume_completes_remaining(monkeypatch, tmp_path):
    import fugu_local
    board = TaskBoard.new(
        "resumable",
        [TodoItem(id="t1", subject="done part", status="completed",
                  result="already done"),
         TodoItem(id="t2", subject="rest part", blocked_by=["t1"])],
        directory=str(tmp_path))
    reloaded = TaskBoard.load(board.board_id, directory=str(tmp_path))
    monkeypatch.setattr(fugu_local, "fugu_answer",
                        lambda q, plan=None, history=None: "finished rest")
    out = fugu_local._run_board(reloaded)
    assert "## done part\nalready done" in out       # 完了済みは再実行されない
    assert "## rest part\nfinished rest" in out
    assert reloaded.all_done() is True


def test_reentrancy_guard_restored_after_run(monkeypatch, tmp_path):
    import fugu_local
    board = TaskBoard.new("q", [TodoItem(id="t1", subject="s")],
                          directory=str(tmp_path))
    monkeypatch.setattr(fugu_local, "fugu_answer",
                        lambda q, plan=None, history=None: "ok")
    fugu_local._run_board(board)
    assert fugu_local._TASKS_ACTIVE is False         # ガードは必ず戻る
