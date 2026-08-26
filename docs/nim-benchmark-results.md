# NIM Backend Benchmark Results (2026-08-08)

Fugu-Local's default and primary mode is 100% local via Ollama (see main README). This
document records a separate, **opt-in** experiment: swapping Ollama for NVIDIA NIM hosted
models to see how the dynamic-MoA design performs once the 8 GB VRAM ceiling is removed,
benchmarked against Fable 5.

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

## Result

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

## Caveats (read before citing the +1)

1. **The win isn't math.** fugu loses AIME 2024 (-2) and AIME 2025 (-1), and ties Fable exactly
   on AIME 2026 — the one set released after Fable's training cutoff, i.e. the cleanest,
   contamination-free comparison. The entire +1 margin comes from JMMLU (+3) and HumanEval
   (+1). The accurate claim is "made up math losses with Japanese MCQ and code," not "beat a
   frontier model at hard math."
2. **How much of the +1 rests on the grader?** Measured, not guessed — `grading_paths.py`
   re-grades both systems with one identical grader and reports which stage credited each
   question (2026-08-20 re-run on the final result set):

   | | raw string | whitespace | normalization | `Fraction` | `math_verify` | code (executed) | total |
   |---|---|---|---|---|---|---|
   | fugu | 157 | 1 | 13 | **0** | **0** | 29 | **200** |
   | Fable 5 | 163 | 1 | 7 | **0** | **0** | 28 | **199** |

   **No answer on either side is credited by sympy-based equivalence.** Every rescue is
   deterministic notation canonicalization (`rac{35}{9}` vs `-35/9`, `30^\circ` vs `30`,
   `	ext{(C)}` vs `C`, LaTeX digit separators), all of it inside MATH-500, and it is applied
   to both sides — Fable is rescued by 8 such questions itself. An earlier version of this
   document reported "strict string matching flips this to fugu 189 / Fable 199 (-10)";
   that figure predates the 2026-08-04 normalizer and the final four questions. Raw
   string-exact scoring on the final set is **fugu 186 / Fable 191 (-5)**, and it penalizes
   both systems for the same reason: MATH-500 ships its gold answers in LaTeX while both
   systems answer in mixed ASCII/LaTeX, so raw equality measures notation, not correctness.
   The +1 remains within run-to-run variance (caveat 5) — but it does not rest on a lenient
   grader.
3. **Cost:** $0 (NVIDIA NIM free tier), 24,758 API requests for the full run.
4. This is an **opt-in cloud experiment**, not the project's default. The headline "100%
   local, no API keys, no cloud" claim in the main README refers to the Ollama backend, not
   this one.
