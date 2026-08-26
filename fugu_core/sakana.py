# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_core.sakana — Sakana AI の TRINITY / Conductor 機構の再実装。

出典(いずれも ICLR 2026・論文のみ公開):

- TRINITY: An Evolved LLM Coordinator (arXiv:2512.04695)
  小型コーディネータが毎ターン「どのモデルに」「どの役(Thinker/Worker/Verifier)を」
  割り当てるかを決め、Verifier が ACCEPT を出した時点で停止する多ターン協調。
- Learning to Orchestrate Agents in Natural Language with the Conductor
  (arXiv:2512.04388) — 指揮者がワーカー間の**通信トポロジ**を設計し、各ワーカーに
  **個別最適化した指示**を書き下ろす。自分自身をワーカーに選ぶと再帰構造になる。

**重要な差分(意図的)**: 論文のコーディネータは学習済みである。TRINITY は
Qwen3-0.6B + 約1万パラメータの線形ヘッドを sep-CMA-ES で進化させたもの、Conductor は
7B を RL で訓練したものだが、**どちらも重みは公開されていない**(Sakana Fugu 製品は
クローズド)。さらに TRINITY の学習済みヘッドは「そのプール7体の並び順」に紐づく出力を
持つため、布陣の違う我々のプールへ移してもそのままでは意味を成さない。
したがってここでは **役割割当とトポロジ設計をプロンプトで行う**。機構は論文どおり、
選択器だけが「進化/RL 済み」ではなく「指揮者 LLM に問う」形になっている。
将来 `pick_fn` を差し替えれば、我々のプールで進化させたヘッドに載せ替えられる
(そのための注入点として `pick_fn` / `plan_fn` を切ってある)。

依存は注入のみ(`runner` / `coordinator`)。fugu_local を import しないので
オフラインで完全にテストできる。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: runner(model, prompt, system=None) -> str。失敗は "__ERROR__..." か例外のいずれでもよい。
Runner = Callable[..., str]

ROLES: Tuple[str, ...] = ("thinker", "worker", "verifier")

#: 役ごとのシステムプロンプト(論文の役割定義に対応)。
ROLE_SYSTEM: Dict[str, str] = {
    "thinker": (
        "You are the THINKER. Analyse the current state of the attempt and propose a "
        "high-level strategy: a decomposition, a different representation, or a critique "
        "of what has been tried. Be concrete about WHAT to do next and WHY. "
        "Do NOT write out the full solution and do NOT state a final answer."),
    "worker": (
        "You are the WORKER. Carry out the next concrete steps of the solution, using the "
        "strategy and any partial work above. Show the work. If the problem is now fully "
        "solved, state the final answer at the very end inside \\boxed{}."),
    "verifier": (
        "You are the VERIFIER. Decide whether the work above is both CORRECT and COMPLETE. "
        "Check the reasoning, not just the plausibility of the answer. "
        "Reply with exactly one word on the first line: ACCEPT if the solution is correct "
        "and complete, or REVISE if anything is wrong, unjustified or missing. "
        "Then give a one-line reason."),
}

_ACCEPT_RE = re.compile(r"^\W*ACCEPT\b", re.I)


def _err(text: Any) -> bool:
    return not text or not str(text).strip() or str(text).startswith("__ERROR__")


def _json_obj(text: str) -> Optional[dict]:
    """出力から最初の JSON オブジェクトを取り出す(失敗は None)。"""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for cand in (text, ):
        try:
            v = json.loads(cand.strip())
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                return v
        except Exception:
            return None
    return None


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " …(略)"


# ============================================================
# TRINITY — 役割割当つき多ターン協調
# ============================================================

def default_pick(pool: Sequence[str], transcript: Sequence[dict], turn: int,
                 max_turns: int) -> Tuple[str, str]:
    """コーディネータが使えない/壊れた時の決定的フォールバック。

    論文の学習済みヘッドの代わりに、Thinker → Worker → Verifier を回し、
    モデルはプールをラウンドロビンする。**必ず何かを返す** — 割当が決まらないことを
    理由に問題を落とさない。最終ターンは必ず Verifier にはしない(Worker の答えを
    残すため)。"""
    order = ("thinker", "worker", "verifier")
    role = order[turn % len(order)]
    if turn == 0:
        role = "thinker"
    return pool[turn % len(pool)], role


def coordinator_pick(pool: Sequence[str], transcript: Sequence[dict], turn: int,
                     max_turns: int, *, coordinator: str, runner: Runner,
                     question: str = "") -> Tuple[str, str]:
    """指揮者 LLM に (モデル, 役) を選ばせる。論文の学習済みヘッドに相当する箇所。

    壊れた JSON・プール外のモデル・未知の役はすべて `default_pick` に落ちる。"""
    hist = "\n".join(
        f"[turn {i + 1}] {t['model']} as {t['role'].upper()}:\n{_clip(t['text'], 700)}"
        for i, t in enumerate(transcript)) or "(まだ何も実行していない)"
    prompt = (
        "You are the COORDINATOR of a pool of LLMs solving one hard problem together.\n"
        "Each turn you choose ONE model from the pool and ONE role for it.\n\n"
        "Roles:\n"
        "  thinker  - propose strategy / decomposition / critique (no final answer)\n"
        "  worker   - execute concrete solution steps\n"
        "  verifier - judge correctness and completeness; answers ACCEPT or REVISE\n\n"
        f"Pool: {', '.join(pool)}\n"
        f"Turn {turn + 1} of at most {max_turns}.\n\n"
        f"Problem:\n{_clip(question, 1500)}\n\n"
        f"Transcript so far:\n{hist}\n\n"
        "Pick the model and role that most advances the solution now. Choose verifier "
        "only once there is a complete candidate solution to check.\n"
        'Return ONLY JSON: {"model": "<one of the pool>", "role": '
        '"thinker|worker|verifier"}')
    try:
        raw = runner(coordinator, prompt, system="Return only the JSON object.",
                     json_mode=True)
    except Exception:
        raw = ""
    obj = None if _err(raw) else _json_obj(raw)
    if obj:
        model, role = str(obj.get("model", "")).strip(), str(obj.get("role", "")).strip().lower()
        if model in pool and role in ROLES:
            return model, role
        # モデル名だけの表記ゆれ(接頭辞落ち等)は救う
        for m in pool:
            if model and (m.endswith(model) or model.endswith(m)):
                if role in ROLES:
                    return m, role
    return default_pick(pool, transcript, turn, max_turns)


def solve(question: str, pool: Sequence[str], *, runner: Runner,
          coordinator: Optional[str] = None, max_turns: int = 5,
          extract: Optional[Callable[[str], Any]] = None,
          pick_fn: Optional[Callable[..., Tuple[str, str]]] = None,
          on_turn: Optional[Callable[[int, str, str, str], None]] = None,
          task_type: str = "math") -> Optional[Dict[str, Any]]:
    """TRINITY 本体。停止条件は論文どおり τ = min{k : role=verifier かつ ACCEPT}。

    戻り値 {"answer", "text", "turns", "accepted", "transcript"}。
    Worker が一度も走らなければ答えは無いので None を返し、呼び出し側は従来経路へ
    落ちる(この機構が既存の解答能力を奪わないための不変条件)。"""
    pool = [m for m in dict.fromkeys(pool) if m]
    if not pool:
        return None
    max_turns = max(1, int(max_turns))
    transcript: List[Dict[str, Any]] = []
    accepted = False

    for turn in range(max_turns):
        if pick_fn is not None:
            model, role = pick_fn(pool, transcript, turn, max_turns)
        elif coordinator:
            model, role = coordinator_pick(pool, transcript, turn, max_turns,
                                           coordinator=coordinator, runner=runner,
                                           question=question)
        else:
            model, role = default_pick(pool, transcript, turn, max_turns)
        # 検証すべき仕事がまだ無いのに verifier を選ぶのは無駄なターン
        if role == "verifier" and not any(t["role"] == "worker" for t in transcript):
            role = "worker"

        body = "\n\n".join(
            f"[{t['role'].upper()} / {t['model']}]\n{t['text']}" for t in transcript)
        prompt = (f"Problem:\n{question}\n\n"
                  + (f"Work so far:\n{body}\n\n" if body else "")
                  + "Now perform your role.")
        try:
            out = runner(model, prompt, system=ROLE_SYSTEM[role])
        except Exception as exc:
            out = f"__ERROR__: {exc}"
        if _err(out):
            # 死んだターンは記録だけして次へ。ターン予算は消費する(無限ループ回避)。
            transcript.append({"model": model, "role": role, "text": "", "error": True,
                               "question": question})
            if on_turn:
                on_turn(turn, model, role, "")
            continue
        transcript.append({"model": model, "role": role, "text": out, "question": question})
        if on_turn:
            on_turn(turn, model, role, out)
        if role == "verifier" and _ACCEPT_RE.match(out.strip()):
            accepted = True
            break

    workers = [t for t in transcript if t["role"] == "worker" and t.get("text")]
    if not workers:
        return None
    text = workers[-1]["text"]
    answer = extract(text) if extract else None
    if answer is None and extract:
        # 最後の Worker から取れなければ、遡って取れる最初のものを使う
        for t in reversed(workers[:-1]):
            answer = extract(t["text"])
            if answer is not None:
                text = t["text"]
                break
    return {"answer": answer, "text": text, "turns": len(transcript),
            "accepted": accepted, "transcript": transcript}


# ============================================================
# Conductor — 通信トポロジ設計 + ワーカー個別指示
# ============================================================

def _validate_topology(obj: Any, pool: Sequence[str], max_workers: int) -> Optional[dict]:
    """トポロジの健全性検査。**先に定義されたワーカーしか入力にできない**規則で
    巡回を構造的に排除する(トポロジカルソート不要・実行順 = 宣言順)。"""
    if not isinstance(obj, dict):
        return None
    ws = obj.get("workers")
    if not isinstance(ws, list) or not ws:
        return None
    seen: List[str] = []
    out: List[dict] = []
    for w in ws[:max_workers]:
        if not isinstance(w, dict):
            return None
        wid = str(w.get("id") or f"w{len(out) + 1}").strip()
        model = str(w.get("model") or "").strip()
        instr = str(w.get("instruction") or "").strip()
        if not wid or wid in seen or model not in pool or not instr:
            return None
        inputs = w.get("inputs") or []
        if not isinstance(inputs, list):
            return None
        inputs = [str(i).strip() for i in inputs]
        if any(i not in seen for i in inputs):        # 前方参照・自己参照・巡回を弾く
            return None
        seen.append(wid)
        out.append({"id": wid, "model": model, "instruction": instr, "inputs": inputs})
    final = str(obj.get("final") or (out[-1]["id"] if out else "")).strip()
    if final not in seen:
        final = out[-1]["id"]
    return {"workers": out, "final": final}


def flat_topology(pool: Sequence[str], question: str, n: int = 3) -> dict:
    """指揮者が使えない時の退避トポロジ: 全員が同じ問題を独立に解き、最後に統合。
    既存の MoA と同型なので、失敗しても能力が落ちないことが保証できる。"""
    ws = [{"id": f"w{i + 1}", "model": m, "inputs": [],
           "instruction": "Solve the problem independently and completely."}
          for i, m in enumerate(list(dict.fromkeys(pool))[:max(1, n)])]
    return {"workers": ws, "final": ws[-1]["id"]}


def plan_topology(question: str, pool: Sequence[str], *, coordinator: str, runner: Runner,
                  max_workers: int = 4, allow_self: bool = True) -> dict:
    """指揮者に通信トポロジと各ワーカーへの個別指示を書かせる(論文の中核)。

    `allow_self=True` なら指揮者自身もワーカー候補に入る — 論文が言う再帰的トポロジ。
    JSON が壊れていれば `flat_topology` に落ちる(絶対に例外を投げない)。"""
    cands = list(dict.fromkeys(list(pool) + ([coordinator] if allow_self else [])))
    prompt = (
        "You are the CONDUCTOR. You do not solve the problem yourself; you design how a "
        "team of LLM agents will solve it.\n\n"
        f"Available workers: {', '.join(cands)}\n"
        f"Use at most {max_workers} workers.\n\n"
        "Design two things:\n"
        "1. The communication topology: which workers run independently, and which ones "
        "read the output of which earlier workers. A worker may only read workers listed "
        "before it.\n"
        "2. A focused, individually tailored instruction for each worker that plays to "
        "what that particular model is good at. The instructions must differ from each "
        "other — do not give everyone the same task.\n\n"
        f"Problem:\n{question}\n\n"
        'Return ONLY JSON:\n'
        '{"workers":[{"id":"w1","model":"<from the list>","instruction":"...",'
        '"inputs":[]},{"id":"w2","model":"...","instruction":"...","inputs":["w1"]}],'
        '"final":"w2"}')
    try:
        raw = runner(coordinator, prompt, system="Return only the JSON object.",
                     json_mode=True)
    except Exception:
        raw = ""
    topo = None if _err(raw) else _validate_topology(_json_obj(raw), cands, max_workers)
    return topo or flat_topology(pool, question, n=min(3, max_workers))


def run_topology(question: str, topo: dict, *, runner: Runner,
                 on_worker: Optional[Callable[[str, str, str], None]] = None
                 ) -> Optional[Dict[str, Any]]:
    """トポロジを宣言順に実行する。各ワーカーは自分の指示 + 入力ワーカーの出力を見る。
    全員が失敗したときだけ None(呼び出し側は従来経路へ)。"""
    outputs: Dict[str, str] = {}
    last_ok: Optional[str] = None
    for w in topo.get("workers", []):
        ctx = "\n\n".join(f"[{i} said]\n{_clip(outputs[i], 4000)}"
                          for i in w["inputs"] if outputs.get(i))
        prompt = (f"Problem:\n{question}\n\n"
                  + (f"Input from other agents:\n{ctx}\n\n" if ctx else "")
                  + f"Your task:\n{w['instruction']}")
        try:
            out = runner(w["model"], prompt,
                         system="You are one agent in a team. Do your assigned task well.")
        except Exception as exc:
            out = f"__ERROR__: {exc}"
        if _err(out):
            continue
        outputs[w["id"]] = out
        last_ok = w["id"]
        if on_worker:
            on_worker(w["id"], w["model"], out)
    if not outputs:
        return None
    final_id = topo.get("final")
    if final_id not in outputs:
        final_id = last_ok
    return {"text": outputs[final_id], "final": final_id,
            "outputs": outputs, "topology": topo}
