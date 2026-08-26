# NIM Backend Benchmark Results (2026-08-08, corrected 2026-08-27)

Fugu-Local's default and primary mode is 100% local via Ollama (see main README). This
document records a separate, **opt-in** experiment: swapping Ollama for NVIDIA NIM hosted
models to see how the dynamic-MoA design performs once the 8 GB VRAM ceiling is removed,
benchmarked against Fable 5.

> **Read this before citing any number here (2026-08-27).** Two claims on this page have been
> corrected, both against us.
>
> 1. **The headline is a tie, not +1.** The 200/210 below is the 2026-08-08 run, which used a
>    base configuration plus two override passes. A *single* configuration run over all 210
>    questions in one serial queue (2026-08-21..23) scores **199/210 — level with Fable 5, not
>    ahead of it**. Cite the tie.
> 2. **The old "-10 under strict grading" was not a like-for-like comparison.** It set fugu's
>    strict score against Fable's *lenient* one. Graded strictly by the same procedure, Fable
>    scores **191**, not 199. See caveat 2.
>
> The original numbers stay on the page. This project has already had to withdraw one headline
> figure — `docs/site/design.html` records the 199/210 union-oracle correction — and the same
> standard applies here.

## Setup

- Backend: `feature/nim-backend` — `FUGU_NIM_PIN=1` for a fixed model lineup, or the default
  auto-selecting profile that live-probes NIM's `/v1/models` catalog and drops dead/retired
  IDs (fixed IDs rot: several models used in earlier runs were retired or started 404'ing
  mid-project).
- Datasets: AIME 2024, AIME 2025, AIME 2026 (released after model training cutoffs, so
  contamination-free), MATH-500, HumanEval, JMMLU (Japanese MMLU) — 210 questions total.
- Grading: self-consistency majority vote with an equivalence-based answer checker, not exact
  string match — see caveat 2 below.
- Regression gate: every claimed win/tie was re-run and required to hold across a 36-question
  regression set on the *same* frozen configuration before being counted. Result: 36/36,
  0 regressions.

## Result — 2026-08-08 run (base configuration + two override passes)

| Dataset | fugu (NIM) | Fable 5 | Diff |
|---|---|---|---|
| AIME 2024 | 28/30 | 30/30 | **-2** |
| AIME 2025 | 29/30 | 30/30 | **-1** |
| AIME 2026 (contamination-free) | 29/30 | 29/30 | ±0 |
| MATH-500 | 49/50 | 49/50 | ±0 |
| HumanEval | 29/30 | 28/30 | **+1** |
| JMMLU | 36/40 | 33/40 | **+3** |
| **Total** | **200/210** | **199/210** | **+1** |

0 unanswered; the 10-question gap from a perfect score is entirely wrong answers.

## Result — 2026-08-21..23 re-run (one configuration, one serial queue, 43.2 h)

This is the figure to cite. Both systems are graded by the same procedure over the
intersection of question IDs, at both grading strictnesses.

| Dataset | fugu lenient | fugu strict | Fable lenient | Fable strict |
|---|---|---|---|---|
| AIME 2024 | 28/30 | 28/30 | 30/30 | 30/30 |
| AIME 2025 | 29/30 | 29/30 | 30/30 | 30/30 |
| AIME 2026 (contamination-free) | 29/30 | 29/30 | 29/30 | 29/30 |
| MATH-500 | 49/50 | **32/50** | 49/50 | **41/50** |
| HumanEval | 28/30 | 28/30 | 28/30 | 28/30 |
| JMMLU | 36/40 | 36/40 | 33/40 | 33/40 |
| **Total** | **199/210** | **182/210** | **199/210** | **191/210** |

**Lenient: 199 vs 199, a tie. Strict: 182 vs 191, -9.** Reproduce with `regrade_strict.py`.

## Caveats (read before citing any number above)

1. **The 2026-08-08 win isn't math.** fugu loses AIME 2024 (-2) and AIME 2025 (-1), and ties
   Fable exactly on AIME 2026 — the one set released after Fable's training cutoff, i.e. the
   cleanest, contamination-free comparison. The entire +1 margin came from JMMLU (+3) and
   HumanEval (+1). The accurate claim was "made up math losses with Japanese MCQ and code,"
   not "beat a frontier model at hard math." On the 2026-08-21..23 re-run there is no margin
   to explain: it is a tie.

2. **What the grader is and isn't doing.** Measured twice, on two different runs, by two
   independently written graders. Both are reported; they disagree in magnitude and neither
   is wrong.

   An earlier version of this document said: *"strict string-exact matching flips this to fugu
   189 / Fable 199 (-10)."* **That comparison was not like-for-like** — it graded fugu strictly
   and Fable leniently. Graded strictly by the same procedure, Fable scores 191, and both
   graders reproduce that number on their own run.

   | grader | data | fugu strict | Fable strict | gap |
   |---|---|---|---|---|
   | `grading_paths.py` (2026-08-20) | 2026-08-08 run | 186 | 191 | **-5** |
   | `regrade_strict.py` (2026-08-26) | 2026-08-21..23 run | 182 | 191 | **-9** |

   Two findings hold on both runs:

   - **Nothing on either side is credited by sympy-based equivalence.** `grading_paths.py`
     reports `Fraction` = 0 and `math_verify` = 0 for both systems. Every rescue is
     deterministic notation canonicalization (`\frac{35}{9}` vs `-35/9`, `30^\circ` vs `30`,
     `\text{(C)}` vs `C`, LaTeX digit separators). The lenient score does not rest on a
     permissive symbolic checker.
   - **The rescues are not one-sided.** Fable is credited by 7-8 formatting rescues of its own.
     "Lenient grading helps fugu more than Fable" does not survive measurement.

   **The lenient scores contain no inflation.** All 420 graded answers (210 x 2) were checked
   for the case that would matter — an answer scored correct that is actually wrong. There are
   **zero** such cases on either side. No score here needs revising downward.

   The gap is entirely in MATH-500; the other five datasets score identically strict and
   lenient for both systems. Of fugu's 18 strict failures there, **17 are formatting and 1 is
   a genuinely wrong answer**. `fugu_canonical.canonicalize_math_answer()` cleans fugu's
   *emitted* answer — it never sees `expected`, and it changes notation, not value — recovering
   8 of the 17 and taking fugu to **190 strict** against Fable's 191.

   MATH-500's own reference answers are internally inconsistent (`\frac` vs `\dfrac`,
   `-13x+3` vs `288 \pi`), so a perfect strict score is unattainable for either system. Parity,
   not perfection, is the meaningful target. Full breakdown: [`strict-grading.md`](strict-grading.md).

3. **Cost:** $0 (NVIDIA NIM free tier), 24,758 API requests for the 2026-08-08 run; the
   2026-08-21..23 re-run took 43.2 h of wall clock on a serial queue.

4. This is an **opt-in cloud experiment**, not the project's default. The headline "100%
   local, no API keys, no cloud" claim in the main README refers to the Ollama backend, not
   this one.
