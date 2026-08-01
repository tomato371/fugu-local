# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""bench_search のオフラインテスト(集計・予算写像・判定・レポート生成)。
実行系(run)はモデルが要るのでここでは触らない。"""
import json

import pytest

import bench_search as BS


# ---------------------------------------------------------------- 予算の写像

def test_budget_knobs_equalize_generation_budget():
    env_b, rounds_b = BS.budget_knobs("baseline", 2)
    assert env_b == {} and rounds_b == 3          # 初回1 + 追加2ラウンド(=10生成)
    env_s, rounds_s = BS.budget_knobs("search", 2)
    assert env_s["FUGU_SEARCH"] == "1"
    assert env_s["FUGU_SEARCH_BUDGET"] == str(BS.BUDGET_UNIT * 2)   # 同じく10生成
    env_m, _ = BS.budget_knobs("mav", 1)
    assert env_m == {"FUGU_MAV": "1"}
    env_sm, _ = BS.budget_knobs("search-multi", 4)
    assert env_sm["FUGU_SEARCH_MULTI"] == "1"
    assert env_sm["FUGU_SEARCH_BUDGET"] == "20"


def test_category_mapping():
    assert BS.category("aime25") == "math"
    assert BS.category("math500") == "math"
    assert BS.category("humaneval") == "code"


# ------------------------------------------------------------------- 集計

def _row(config, mult, correct, category="math", gen=6, verify=5, secs=100,
         tree=None):
    return {"id": "x", "config": config, "mult": mult, "correct": correct,
            "category": category, "calls_gen": gen, "calls_verify": verify,
            "seconds": secs, "chars_out": 1000, "tree": tree}


def test_aggregate_accuracy_and_categories():
    rows = [_row("baseline", 1, True), _row("baseline", 1, False),
            _row("baseline", 1, True, category="code"),
            _row("search", 1, True, tree={"nodes": 5, "width": 3, "depth": 2,
                                          "best_depth": 1, "score": 0.8})]
    cells = BS.aggregate(rows)
    b = cells[("baseline", 1)]
    assert b["n"] == 3 and b["acc"] == pytest.approx(2 / 3)
    assert b["cats"]["math"]["acc"] == pytest.approx(0.5)
    assert b["cats"]["code"]["acc"] == 1.0
    s = cells[("search", 1)]
    assert s["tree"]["width"] == 3 and s["tree"]["depth"] == 2


# ------------------------------------------------------------------- 判定

def _cells(pairs, tree=None):
    """{(config, mult): acc} から aggregate 相当の cells を作る。"""
    out = {}
    for (config, mult), acc in pairs.items():
        out[(config, mult)] = {"n": 20, "acc": acc, "gen": 6.0, "verify": 5.0,
                               "secs": 100.0, "chars": 1000.0, "cats": {}}
        if config == "search" and tree:
            out[(config, mult)]["tree"] = tree
    return out


def test_verdict_adopts_when_search_wins_at_every_budget():
    cells = _cells({("baseline", 1): 0.4, ("search", 1): 0.5,
                    ("baseline", 2): 0.45, ("search", 2): 0.6,
                    ("baseline", 4): 0.5, ("search", 4): 0.7})
    v = BS.verdict(cells)
    assert v.startswith("採用")


def test_verdict_rejects_and_diagnoses_width_heavy_tree():
    cells = _cells({("baseline", 1): 0.6, ("search", 1): 0.4},
                   tree={"width": 8.0, "depth": 1.5, "nodes": 9.0})
    v = BS.verdict(cells)
    assert v.startswith("不採用")
    assert "幅ばかり" in v and "分散が小さすぎ" in v


def test_verdict_rejects_and_diagnoses_depth_heavy_tree():
    cells = _cells({("baseline", 1): 0.6, ("search", 1): 0.4},
                   tree={"width": 1.0, "depth": 5.0, "nodes": 6.0})
    v = BS.verdict(cells)
    assert "深さばかり" in v and "悲観的" in v


def test_verdict_notes_small_sample():
    cells = _cells({("baseline", 1): 0.4, ("search", 1): 0.6})
    for c in cells.values():
        c["n"] = 3
    assert "標本を増やす" in BS.verdict(cells)


def test_verdict_without_comparable_cells():
    assert BS.verdict({}).startswith("判定不能")
    only_search = _cells({("search", 1): 0.5})
    assert BS.verdict(only_search).startswith("判定不能")


def test_verdict_baseline_win_is_reported_as_such():
    """ベンチを良く見せない: baseline が勝ったら不採用と断定する。"""
    cells = _cells({("baseline", 1): 0.8, ("search", 1): 0.8,
                    ("baseline", 4): 0.9, ("search", 4): 0.9},
                   tree={"width": 2.0, "depth": 2.0, "nodes": 5.0})
    v = BS.verdict(cells)
    assert v.startswith("不採用")          # 同点どまり(最大予算で上回らない)


# ------------------------------------------------------------------ レポート

def test_reports_are_written_and_self_contained(tmp_path):
    rows = [_row("baseline", 1, True), _row("baseline", 1, False),
            _row("search", 1, True,
                 tree={"nodes": 5, "width": 3, "depth": 2, "best_depth": 1,
                       "score": 0.8}),
            _row("search", 1, True, category="code",
                 tree={"nodes": 4, "width": 2, "depth": 2, "best_depth": 2,
                       "score": 0.9})]
    cells = BS.aggregate(rows)
    md = tmp_path / "report.md"
    html = tmp_path / "dash.html"
    BS.write_md(cells, rows, path=md)
    BS.write_html(cells, rows, path=html)

    md_text = md.read_text(encoding="utf-8")
    assert "| baseline | 1x | 2 | 0.50 " in md_text
    assert "採用" in md_text or "不採用" in md_text or "判定不能" in md_text

    html_text = html.read_text(encoding="utf-8")
    assert "https://" not in html_text.replace("https://hooks", "")  # 外部依存なし
    assert "prefers-color-scheme: dark" in html_text                 # テーマ対応
    payload = html_text.split("const DATA = ", 1)[1]
    data = json.loads(payload[: payload.index(";\n")])
    assert data["rows"] == 4
    assert any(c["config"] == "search" and c["tree"] for c in data["cells"])
