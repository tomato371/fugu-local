# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""bench_search — AB-MCTS 探索 / MAV vs 既存再帰ラウンドの同一予算 A/B 評価。

比較する 4 構成(すべて同じ固定 MoA プラン・SC 無効で、エスカレーション機構だけを
入れ替える — ルーティングの揺らぎを排して「再帰 vs 探索」を分離して測る):

  baseline     : 現行の council + Critic 却下 → 線形再帰ラウンド
  mav          : baseline + best-of-N 多検証者スコアリング (FUGU_MAV=1)
  search       : Critic 却下 → AB-MCTS 探索 (FUGU_SEARCH=1)
  search-multi : search + Multi-LLM 選択 (FUGU_SEARCH_MULTI=1)

予算の定義(全構成で同一): 予算 x とは「初回ラウンド後に追加で許す生成 LLM 呼び出し
数 = x × BUDGET_UNIT」。線形版は追加ラウンド数(1 ラウンド ≈ 提案3+統合1+批評1 =
BUDGET_UNIT)で消費し、探索版は FUGU_SEARCH_BUDGET で同数を消費する。
検証者呼び出し(小型モデル)は生成予算に含めず別カラムで正直に報告する。

  python bench_search.py run --dataset aime25 --configs baseline,search --mults 1,2,4 --limit 6
  python bench_search.py report        # docs/search-benchmark.md + .html を生成

1 問ずつ JSONL 追記・再開可能(bench_fugu と同じ流儀)。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

import bench_fugu as B
import fugu_local as f
import fugu_search

BUDGET_UNIT = 5              # 追加ラウンド 1 回ぶんの生成呼び出し(提案3+統合1+批評1)
RESULTS_DIR = B.BENCH_DIR / "search_results"
DOCS_MD = Path("docs/search-benchmark.md")
DOCS_HTML = Path("docs/search-benchmark.html")

CONFIGS = {
    "baseline": {},
    "mav": {"FUGU_MAV": "1"},
    "search": {"FUGU_SEARCH": "1"},
    "search-multi": {"FUGU_SEARCH": "1", "FUGU_SEARCH_MULTI": "1"},
}

#: データセット → カテゴリ(検証が厳密な数学 vs コード)
def category(dataset):
    return "code" if dataset.startswith("humaneval") else "math"


# ---------------------------------------------------------------- 呼び出し計測

class CallCounter:
    """f.ask をラップして生成/検証の呼び出し数と文字数を数える。
    label が verify-* なら検証者(小型モデル)、それ以外は生成として分類。"""

    def __init__(self):
        self.gen = self.verify = 0
        self.chars_out = 0

    def install(self):
        self._orig = f.ask

        def counting_ask(model, messages, temperature=0.2, **kw):
            label = str(kw.get("label") or "")
            out = self._orig(model, messages, temperature, **kw)
            if label.startswith("verify"):
                self.verify += 1
            else:
                self.gen += 1
            self.chars_out += len(out or "")
            return out

        f.ask = counting_ask
        return self

    def uninstall(self):
        f.ask = self._orig


class SearchCapture:
    """fugu_search.search をラップして探索木の形(幅/深さ)を記録する。"""

    def __init__(self):
        self.shapes = []

    def install(self):
        self._orig = fugu_search.search

        def capturing(*a, **kw):
            res = self._orig(*a, **kw)
            self.shapes.append({"nodes": res.nodes, "width": res.root_width,
                                "depth": res.max_depth,
                                "best_depth": res.best_depth,
                                "score": res.score})
            return res

        fugu_search.search = capturing
        return self

    def uninstall(self):
        fugu_search.search = self._orig


# ------------------------------------------------------------------- 実行

def budget_knobs(config, mult):
    """(env 追加分, MAX_ROUNDS) — 全構成で追加生成予算を BUDGET_UNIT×mult に揃える。"""
    env = dict(CONFIGS[config])
    if "FUGU_SEARCH" in env:
        # 探索版: 追加ラウンドの代わりに探索が同数の生成呼び出しを使う
        env["FUGU_SEARCH_BUDGET"] = str(BUDGET_UNIT * mult)
        max_rounds = 1 + mult          # 探索が失敗したときの線形フォールバック上限
    else:
        max_rounds = 1 + mult          # 初回 1 + 追加 mult ラウンド(= mult×5 生成)
    return env, max_rounds


def fixed_plan(item):
    """Conductor を固定した MoA プラン(bench_fugu の run_moa_old / run_coder と同じ
    流儀)。ルーティングの揺らぎを排し、エスカレーション機構だけを比較する。"""
    return {"mode": "moa", "task_type": item["task_type"],
            "selected_proposers": f.PROPOSERS[:3], "rounds": 1,
            "use_image_generation": False, "image_only": False,
            "make_pptx": False, "search_required": False,
            "reason": "bench_search", "_fallback": False}


def apply_council(council, aggregator=None):
    """council/aggregator を一時的に差し替え、復元関数を返す。

    フルの council (20b〜35b) は 8GB GPU で 1 問 1 時間級になるため、軽量モデルに
    落として全セルを実測する用途(--council)。4 構成すべてに同条件で適用されるので
    「線形再帰 vs 探索」の比較としては公平なまま — ただしフル構成の代表値ではない
    ことをレポートに明記する(結果行にも council を記録する)。"""
    if not council:
        return lambda: None
    saved = (f.PROPOSERS, f.AGGREGATOR)
    f.PROPOSERS = list(council)
    f.AGGREGATOR = aggregator or council[-1]

    def restore():
        f.PROPOSERS, f.AGGREGATOR = saved
    return restore


def run_one(item, config, mult):
    env, max_rounds = budget_knobs(config, mult)
    saved_env = {k: os.environ.get(k) for k in
                 ("FUGU_MAV", "FUGU_SEARCH", "FUGU_SEARCH_MULTI",
                  "FUGU_SEARCH_BUDGET")}
    saved = (f.MAX_ROUNDS, f.MAX_ROUNDS_CODE, f.SC_ENABLED)
    counter, capture = CallCounter().install(), SearchCapture().install()
    try:
        for k in saved_env:
            os.environ.pop(k, None)
        os.environ.update(env)
        f.MAX_ROUNDS = f.MAX_ROUNDS_CODE = max_rounds
        f.SC_ENABLED = False           # SC は math を横取りする — MoA 経路に固定
        t0 = time.time()
        text = f.fugu_answer(item["question"], plan=fixed_plan(item)) or ""
        seconds = round(time.time() - t0, 1)
        correct, got, note = B.grade_item(item, answer_text=text)
        return {"correct": bool(correct), "got": got, "note": note,
                "seconds": seconds, "calls_gen": counter.gen,
                "calls_verify": counter.verify, "chars_out": counter.chars_out,
                "tree": capture.shapes[-1] if capture.shapes else None,
                "answer_text": text[-2000:]}
    finally:
        counter.uninstall()
        capture.uninstall()
        f.MAX_ROUNDS, f.MAX_ROUNDS_CODE, f.SC_ENABLED = saved
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def results_path(dataset, config, mult):
    return RESULTS_DIR / f"{dataset}__{config}__{mult}x.jsonl"


def load_done(dataset, config, mult):
    p = results_path(dataset, config, mult)
    done = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def run(dataset, configs, mults, limit=None, offset=0, council=None,
        aggregator=None):
    items = B.load_items(dataset)
    rng = random.Random(42)            # bench_fugu と同じ決定的シャッフル
    rng.shuffle(items)
    items = items[offset:offset + limit] if limit else items[offset:]
    if not f.setup():
        raise SystemExit("fugu setup 失敗（Ollama を確認）")
    f.SHOW_PLAN = f.SHOW_PROPOSALS = False
    restore = apply_council(council, aggregator)
    if council:
        print(f"[bench_search] 軽量council: {council} / "
              f"aggregator={f.AGGREGATOR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for mult in mults:
        for config in configs:
            done = load_done(dataset, config, mult)
            todo = [it for it in items if it["id"] not in done]
            print(f"\n[bench_search] {dataset} × {config} × {mult}x: "
                  f"{len(todo)} 問（済 {len(items) - len(todo)}）")
            for k, it in enumerate(todo, 1):
                print(f"=== [{k}/{len(todo)}] {it['id']} ({config}/{mult}x) ===")
                rec = {"id": it["id"], "dataset": dataset, "config": config,
                       "mult": mult, "expected": it["answer"],
                       "category": category(dataset),
                       "council": list(council) if council else None,
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
                try:
                    rec.update(run_one(it, config, mult))
                except KeyboardInterrupt:
                    print("[bench_search] 中断（結果は保存済み・再実行で再開）")
                    restore()
                    raise
                except Exception as e:
                    rec.update({"correct": False,
                                "error": f"{type(e).__name__}: {e}"})
                with results_path(dataset, config, mult).open(
                        "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"    -> {'OK' if rec.get('correct') else 'NG'} "
                      f"gen={rec.get('calls_gen')} verify={rec.get('calls_verify')} "
                      f"({rec.get('seconds')}s)")
    restore()


# ------------------------------------------------------------------- 集計

def load_all_rows():
    rows = []
    if RESULTS_DIR.exists():
        for p in sorted(RESULTS_DIR.glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def aggregate(rows):
    """(config, mult) → {n, acc, per-category acc, 平均コスト, 木の形}"""
    cells = {}
    for r in rows:
        key = (r["config"], r["mult"])
        c = cells.setdefault(key, {
            "n": 0, "ok": 0, "gen": 0, "verify": 0, "secs": 0.0, "chars": 0,
            "cats": {}, "trees": []})
        c["n"] += 1
        c["ok"] += int(bool(r.get("correct")))
        c["gen"] += r.get("calls_gen") or 0
        c["verify"] += r.get("calls_verify") or 0
        c["secs"] += r.get("seconds") or 0.0
        c["chars"] += r.get("chars_out") or 0
        cat = c["cats"].setdefault(r.get("category", "?"), {"n": 0, "ok": 0})
        cat["n"] += 1
        cat["ok"] += int(bool(r.get("correct")))
        if r.get("tree"):
            c["trees"].append(r["tree"])
    out = {}
    for key, c in cells.items():
        n = c["n"]
        entry = {
            "n": n, "acc": c["ok"] / n,
            "gen": c["gen"] / n, "verify": c["verify"] / n,
            "secs": c["secs"] / n, "chars": c["chars"] / n,
            "cats": {k: {"n": v["n"], "acc": v["ok"] / v["n"]}
                     for k, v in c["cats"].items()},
        }
        if c["trees"]:
            t = c["trees"]
            entry["tree"] = {
                "width": sum(x["width"] for x in t) / len(t),
                "depth": sum(x["depth"] for x in t) / len(t),
                "nodes": sum(x["nodes"] for x in t) / len(t),
            }
        out[key] = entry
    return out


def verdict(cells):
    """レポート末尾の 1 行断定。同一予算で search vs baseline を比較する。"""
    mults = sorted({m for (c, m) in cells if c in ("baseline", "search")
                    and ("baseline", m) in cells and ("search", m) in cells})
    if not mults:
        return "判定不能: baseline と search の同一予算セルがまだ揃っていない。"
    wins = [m for m in mults
            if cells[("search", m)]["acc"] > cells[("baseline", m)]["acc"]]
    ties = [m for m in mults
            if cells[("search", m)]["acc"] == cells[("baseline", m)]["acc"]]
    top = mults[-1]
    n_min = min(cells[("search", m)]["n"] for m in mults)
    small = f"（ただし各セル n={n_min} — 有意性を語るには標本を増やすこと）" \
        if n_min < 10 else ""
    if len(wins) + len(ties) == len(mults) and top in wins:
        return (f"採用: 同一予算で search が baseline を上回った"
                f"（勝ち {len(wins)}/{len(mults)} 予算点、最大予算 {top}x でも優位）"
                + small)
    # 不採用: 木の形から原因を述べる
    shape = cells.get(("search", top), {}).get("tree")
    cause = ""
    if shape:
        if shape["width"] >= 2 * shape["depth"]:
            cause = ("原因の仮説: 幅ばかり広がっている"
                     f"（width {shape['width']:.1f} vs depth {shape['depth']:.1f}）"
                     "= 報酬(検証スコア)の分散が小さすぎて精緻化が選ばれない。")
        elif shape["depth"] >= 2 * shape["width"]:
            cause = ("原因の仮説: 深さばかり進んでいる"
                     f"（depth {shape['depth']:.1f} vs width {shape['width']:.1f}）"
                     "= GEN の事前分布が悲観的すぎて幅の探索が起きない。")
        else:
            cause = (f"木の形は均衡（width {shape['width']:.1f} / "
                     f"depth {shape['depth']:.1f}）— 報酬関数の識別力を疑うこと。")
    return (f"不採用: 同一予算で search は baseline を上回らなかった"
            f"（勝ち {len(wins)}/{len(mults)} 予算点）。{cause}{small}")


# ------------------------------------------------------------------- 出力

def write_md(cells, rows, path=DOCS_MD):
    councils = sorted({tuple(r["council"]) for r in rows if r.get("council")})
    council_note = ""
    if councils:
        pretty = " / ".join(", ".join(c) for c in councils)
        council_note = (
            f"\n\n**⚠ 軽量council での実測**: {pretty}。フル構成(20b〜35b)は "
            "8GB GPU で 1 問 1 時間級となり行列の完走が現実的でないため、全 4 構成に"
            "同一の軽量 council を適用して比較した(構成間の比較としては公平だが、"
            "フル構成での絶対値の代表ではない)。")
    lines = ["# AB-MCTS 探索 A/B ベンチマーク", "",
             "同一の追加生成予算（1x = 追加 5 生成呼び出し）で、線形再帰ラウンド"
             "(baseline) と AB-MCTS 探索(search) を比較する。検証者呼び出し"
             "（小型モデル）は生成予算外・別カラム。トークン数の代理として出力文字数"
             "を記録（正確なトークン計測は FUGU_PROFILE 系の整備後）。"
             + council_note, "",
             f"総試行: {len(rows)} 行", "",
             "| 構成 | 予算 | n | 正答率 | math | code | 生成/問 | 検証/問 "
             "| 秒/問 | 木(幅×深さ) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (config, mult) in sorted(cells, key=lambda k: (k[1], k[0])):
        c = cells[(config, mult)]
        cats = c["cats"]
        tree = c.get("tree")
        tree_s = f"{tree['width']:.1f}×{tree['depth']:.1f}" if tree else "—"

        def _cat(name):
            e = cats.get(name)
            return f"{e['acc']:.2f}" if e else "—"

        lines.append(
            f"| {config} | {mult}x | {c['n']} | {c['acc']:.2f} "
            f"| {_cat('math')} | {_cat('code')} "
            f"| {c['gen']:.1f} | {c['verify']:.1f} | {c['secs']:.0f} "
            f"| {tree_s} |")
    lines += ["", "ダッシュボード: [search-benchmark.html](search-benchmark.html)",
              "", "## 判定", "", verdict(cells), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path}")


def write_html(cells, rows, path=DOCS_HTML):
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "rows": len(rows),
        "verdict": verdict(cells),
        "cells": [{"config": c, "mult": m, **v} for (c, m), v in
                  sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0]))],
    }
    html = _DASHBOARD_TEMPLATE.replace("/*DATA*/null", json.dumps(
        payload, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[report] {path}")


_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AB-MCTS 探索ベンチ</title>
<style>
.viz-root{color-scheme:light;
 --surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz-root{
 color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;
 --muted:#898781;--grid:#2c2c2a;--axis:#383835;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;}}
:root[data-theme="dark"] .viz-root{
 color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;
 --muted:#898781;--grid:#2c2c2a;--axis:#383835;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;}
body{margin:0}
.viz-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
 background:var(--page);color:var(--ink);padding:24px;min-height:100vh}
.viz-root h1{font-size:20px;margin:0 0 4px}
.viz-root .sub{color:var(--ink-2);font-size:13px;margin-bottom:20px}
.card{background:var(--surface-1);border:1px solid var(--grid);border-radius:10px;
 padding:16px;margin-bottom:16px;max-width:960px}
.card h2{font-size:14px;margin:0 0 10px;color:var(--ink-2);font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink-2);
 margin-bottom:8px}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:3px;
 margin-right:5px;vertical-align:-1px}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{color:var(--muted);font-weight:500;text-align:right;padding:6px 8px;
 border-bottom:1px solid var(--axis)}
.tbl td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid);
 font-variant-numeric:tabular-nums}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.verdict{font-size:14px;padding:12px 16px;border-left:3px solid var(--s1);
 background:var(--surface-1)}
svg text{fill:var(--muted);font-size:11px;font-family:inherit}
svg .lbl{fill:var(--ink-2);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);
 border:1px solid var(--axis);border-radius:6px;padding:6px 9px;font-size:12px;
 color:var(--ink);display:none;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.wrap{overflow-x:auto}
</style></head><body><div class="viz-root">
<h1>AB-MCTS 探索 — 予算対精度スケーリング</h1>
<div class="sub" id="sub"></div>
<div class="card"><h2>正答率 × 追加生成予算（1x = +5 生成呼び出し/問）</h2>
<div class="legend" id="legend"></div>
<div class="wrap"><svg id="curve" width="640" height="300"
 viewBox="0 0 640 300" role="img" aria-label="予算対精度の折れ線"></svg></div></div>
<div class="card"><h2>セル別の実測値（表ビュー）</h2>
<div class="wrap"><table class="tbl" id="table"></table></div></div>
<div class="card verdict" id="verdict"></div>
<div class="tip" id="tip"></div>
<script>
const DATA = /*DATA*/null;
const ORDER = ["baseline","mav","search","search-multi"];
const COLOR = {baseline:"var(--s1)",mav:"var(--s2)",search:"var(--s3)","search-multi":"var(--s4)"};
document.getElementById("sub").textContent =
  `生成 ${DATA.generated} · ${DATA.rows} 試行`;
document.getElementById("verdict").textContent = `判定: ${DATA.verdict}`;
const mults=[...new Set(DATA.cells.map(c=>c.mult))].sort((a,b)=>a-b);
const svg=document.getElementById("curve"), tip=document.getElementById("tip");
const L=52,R=90,T=16,BT=262,W=640;
const x=m=>{const i=mults.indexOf(m);return L+(W-L-R)*(mults.length<2?0.5:i/(mults.length-1));};
const y=a=>BT-(BT-T)*a;
let g="";
for(const t of [0,0.25,0.5,0.75,1])
  g+=`<line x1="${L}" x2="${W-R}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/>`+
     `<text x="${L-8}" y="${y(t)+4}" text-anchor="end">${(t*100).toFixed(0)}%</text>`;
g+=`<line x1="${L}" x2="${W-R}" y1="${BT}" y2="${BT}" stroke="var(--axis)"/>`;
for(const m of mults) g+=`<text x="${x(m)}" y="${BT+18}" text-anchor="middle">${m}x</text>`;
const legend=document.getElementById("legend");
for(const cfg of ORDER){
  const pts=mults.map(m=>DATA.cells.find(c=>c.config===cfg&&c.mult===m))
                 .map((c,i)=>c?[x(mults[i]),y(c.acc),c]:null).filter(Boolean);
  if(!pts.length) continue;
  legend.insertAdjacentHTML("beforeend",
    `<span><span class="sw" style="background:${COLOR[cfg]}"></span>${cfg}</span>`);
  if(pts.length>1)
    g+=`<polyline fill="none" stroke="${COLOR[cfg]}" stroke-width="2" `+
       `points="${pts.map(p=>p[0]+","+p[1]).join(" ")}"/>`;
  for(const p of pts)
    g+=`<circle cx="${p[0]}" cy="${p[1]}" r="4.5" fill="${COLOR[cfg]}" `+
       `stroke="var(--surface-1)" stroke-width="2" data-cfg="${cfg}" `+
       `data-mult="${p[2].mult}" data-acc="${p[2].acc}" data-n="${p[2].n}"/>`;
  const last=pts[pts.length-1];
  g+=`<text class="lbl" x="${last[0]+10}" y="${last[1]+4}">${cfg}</text>`;
}
svg.innerHTML=g;
svg.addEventListener("mousemove",e=>{
  const c=e.target.closest("circle");
  if(!c){tip.style.display="none";return;}
  tip.style.display="block";
  tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+14)+"px";
  tip.textContent=`${c.dataset.cfg} ${c.dataset.mult}x — 正答率 `+
    `${(100*+c.dataset.acc).toFixed(0)}% (n=${c.dataset.n})`;});
svg.addEventListener("mouseleave",()=>tip.style.display="none");
const tbl=document.getElementById("table");
tbl.innerHTML="<tr><th>構成</th><th>予算</th><th>n</th><th>正答率</th>"+
 "<th>math</th><th>code</th><th>生成/問</th><th>検証/問</th><th>秒/問</th>"+
 "<th>幅×深さ</th></tr>"+DATA.cells.map(c=>{
  const f=v=>v==null?"—":(typeof v==="number"?v.toFixed(2):v);
  const cat=k=>c.cats&&c.cats[k]?c.cats[k].acc.toFixed(2):"—";
  const tr=c.tree?`${c.tree.width.toFixed(1)}×${c.tree.depth.toFixed(1)}`:"—";
  return `<tr><td>${c.config}</td><td>${c.mult}x</td><td>${c.n}</td>`+
   `<td>${c.acc.toFixed(2)}</td><td>${cat("math")}</td><td>${cat("code")}</td>`+
   `<td>${c.gen.toFixed(1)}</td><td>${c.verify.toFixed(1)}</td>`+
   `<td>${c.secs.toFixed(0)}</td><td>${tr}</td></tr>`;}).join("");
</script></div></body></html>
"""


def report():
    rows = load_all_rows()
    if not rows:
        raise SystemExit("結果がまだ無い。先に `python bench_search.py run ...`")
    cells = aggregate(rows)
    write_md(cells, rows)
    write_html(cells, rows)
    print("\n" + verdict(cells))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--configs", default="baseline,mav,search,search-multi")
    p_run.add_argument("--mults", default="1,2,4")
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--offset", type=int, default=0)
    p_run.add_argument("--council", default=None,
                       help="カンマ区切りで council を軽量モデルに差し替える"
                            "(4 構成すべてに同一適用。先頭 = 探索の既定生成モデル)")
    p_run.add_argument("--aggregator", default=None,
                       help="--council 時の統合モデル(既定: council の末尾)")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        for c in configs:
            if c not in CONFIGS:
                raise SystemExit(f"未知の構成: {c} (choices: {list(CONFIGS)})")
        mults = [int(m) for m in args.mults.split(",")]
        council = [m.strip() for m in args.council.split(",")] \
            if args.council else None
        run(args.dataset, configs, mults, limit=args.limit, offset=args.offset,
            council=council, aggregator=args.aggregator)
    else:
        report()


if __name__ == "__main__":
    main()
