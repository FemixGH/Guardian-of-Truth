# Solution write-up

## 1. Task

- **Input:** one row = `prompt` (everything the agent saw before its turn: a `<policy>`, the `[AVAILABLE TOOLS]`
  schema, the dialogue, earlier tool calls and outputs) + `response` (the agent turn: text and/or tool calls).
  The data is τ²-bench trajectories (airline, retail, telecom, banking_knowledge). User and agent utterances are
  translated into Russian, while policies, tool JSON and tool outputs stay in English.
- **Output:** 1 if the turn contains a *contextual hallucination*, else 0. Metric: F1 on class 1.
- **Test:** public 548 rows, private 543 rows (median ~32K characters, max 232K), ~24% positives.
  The final ranking uses the private set.
- **Runtime:** 1× A100 80 GB, 30 minutes per run, archive under 40 GB, no network, platform vLLM 0.19.1 with
  no `pip install` of extra dependencies.
- **Labelled data provided:** `valid.parquet`, only 46 rows (23/23) with explanations. The organisers state that
  it does not match the test distribution.

## 2. What the labels count

The explanations in `valid.parquet` show a strict, procedure-oriented labeller that counts **only the new
errors of the turn**:

- **Fabricated or wrong arguments**, for example a zip code taken from the digits of an e-mail, or a sum that
  includes already-flown reservations.
- **A non-existent tool, or a discoverable tool called without `unlock_discoverable_agent_tool`.**
- **Policy violations:** a return *and* an exchange on one order, a cancellation with no condition met, or
  unlocking a tool the user is not eligible for (for example, utilisation 75% against a 70% limit).
- **False refusals and premature transfers:** "I can't do this, let me transfer you" while a tool could do it,
  or "the KB has no information" while a search tool was never tried.
- **Claimed actions with no matching call:** "I re-sent the reset and ran a diagnostic" when the only call is a
  transfer.

The following are **not** errors:
- several tool calls in one turn;
- ordinary lookups;
- procedural questions such as asking for an ID or a confirmation;
- generic explanations;
- errors from earlier turns that the turn merely carries over.

## 3. System (final version `s11o`)

See the README for the pipeline diagram. Design choices that mattered:

| Component | Why |
|---|---|
| Deterministic hard flags | Schema, unlock and repeat-failure errors are certain, so the LLM is not asked about them. ~3% of rows, precision ~0.6-0.9 depending on the rule. |
| Checker notes in the judge prompt | The judge sees what the parser found (for example "this argument value never appears in the context") and verifies it. |
| Domain checklists | The biggest single gain (sub7 → sub8). The judge otherwise mis-applies generic "be helpful" intuitions to strict policies. |
| Reasoning, then a verdict with quoted evidence | Single-token judging (the organisers' baseline) reaches F1 0.49. Reasoning reaches 0.61. Quoting source lines (v3m) cut text-reply false alarms. |
| Context compression | Keeps the policy, the schema, the recent turns and the tool-output lines that contain values the turn states. A plain cut loses the evidence. |
| Stop at the first verdict | 43% of reasoning tokens came after the first verdict, which changed in only 0.4% of rows. |
| Second prompt only on clean rows (OR) | Adds recall cheaply. A third pass was tested and rejected. |
| Confirmation by majority of 3 at T=0.6 | Identical runs at T=0 differ by ±0.01-0.02 F1. Re-asking the flagged rows removes the noise flags. |
| Global deadline + fallback | The run always finishes within the limit, and a valid output exists from the first second. |

## 4. Evaluation, and how reliable it was

Only F1 is shown per submission, and at most 10 submissions were allowed, so we built offline proxies:

| Set | Size | Labels | Verdict |
|---|---|---|---|
| `valid.parquet` | 46 | organisers | Too small, and a different distribution. Used only as a sanity check. |
| "Early turns" | 385 (+46 valid = "t431") | blind labels by Claude on earlier agent turns found inside the test prompts | Separates weak configs from strong ones, but **did not rank the strong ones** (Spearman ~0.4 against the public LB). |
| AgentProcessBench (translated) | ~1,000 | human step-level labels from τ²-bench | A different label policy. Useful for false-positive-rate trends only. |
| External GLM-labelled set ([KarimovDev](https://github.com/KarimovDev/guardian-of-truth-dataset-2026)) | 664 (576 natural) | GLM | **The best proxy:** sub11-like version F1 0.60 on the natural rows against 0.61 public. |

We re-ran 7 past submissions end to end and compared them with their public scores. The proxy F1 has a
standard error of ~±0.03, and so does the public F1 itself on ~550 rows. So differences under ~0.03 between
strong configs could not be resolved, and we stopped accepting changes that were not clearly significant.

## 5. Submission history (public leaderboard)

| # | Change | F1 | P | R |
|---|---|---|---|---|
| — | "All ones" probe | 0.384 | | |
| 1 | Judge v2 + v1 (OR) + hard flags | 0.508 | 0.72 | 0.39 |
| 3 | Fixed quota of flags, ranked by a single-token score | 0.390 | | |
| 4 | sub1 + shortest-first + 24K context + watchdog | 0.512 | 0.65 | 0.42 |
| 5 | Prompts v3b + v2 ("parallel calls allowed", "asking for IDs is procedure") | 0.599 | 0.59 | 0.61 |
| 7 | + 4th pass at T=0.7 | 0.592 | | |
| 8 | Domain rules (v3f) + v2 + trace hints | 0.602 | 0.61 | 0.60 |
| **11** | + context compression, policy hard checks, numbers specialist | **0.612** | 0.62 | 0.61 |
| cand12 | v3f only, stop-at-verdict, veto | 0.576 | | |

After sub11, versions s11q → s11v → s11n → s11o added: the confirmation stage, the 3 high-precision rules in
`rules2.py`, the v3m evidence-quoting prompt, and clarifications from the error analysis on the GLM-labelled set.

| Version | GLM natural F1 | t431 early turns F1 | valid F1 |
|---|---|---|---|
| s11v | 0.599 | 0.820 | 1.000 |
| s11n | 0.701 | 0.820 | 0.930 |
| **s11o** | **0.771** | **0.827** | **0.978** |

The s11o − s11n difference on the natural rows is +0.070 (95% CI +0.018 to +0.131, bootstrap over whole dialogues).

## 6. What did not work

Every item below was measured end to end and rejected on the numbers:

- **Other judge models.**
  - Qwen3.8-27B-FP8: 6-7× slower, more false positives.
  - gpt-oss-20b: more false positives, and its unique flags were all wrong.
  - Gemma 4 26B-A4B as a second stage: different errors, but no robust gain within the time budget.
  - Fine-tuned Qwen3.6 variants (several community distills): all flag more and judge more loosely.
  - K2-Horizon-MoVA-36B-A4B: the best single judge we found in BF16, but it is not supported by the platform's
    vLLM 0.19.1. We ported it (`experiments/k2port`), but the int4 and int8 quantisations needed to fit in
    40 GB lost its advantage.
- **Training.** A LoRA classifier and GSPO/SFT on the judge (`experiments/lora`) with AgentProcessBench and
  synthetic errors. With ~190 informative rows, synthetic errors that were too easy, and about one effective
  GSPO step, there was no robust gain.
- **More passes.**
  - Temperature passes: their additions were ~23% precise on test.
  - A third prompt: as many false flags as real ones.
  - OR of two full pipelines: a gain, but it needs 2× the time.
- **Ranking by score.**
  - A fixed quota by single-token score.
  - A baseline-probability gate with re-asking.
  - BM25 evidence verification.

  Our reasoning judge already covers what these find, and the extra flags were below the precision needed
  (~0.5) to raise F1.
- **Prompt variants.** Few-shot from valid explanations, a structured claims table, an expected-turn prompt,
  and a list-format output (also too slow) were all within noise.
- **Longer context (160K characters):** better on long banking rows offline, but scored 0.576 on public.
  Mixed with other changes, so the result is not conclusive.

## 7. Lessons

1. **The label policy matters more than the model.** The largest gains came from writing down what this
   labeller counts: the domain checklists and the "new errors only / procedure is fine" rules.
2. **Measure noise before accepting a change.** A T=0 judge with batching is not deterministic. Two identical
   runs differ by about as much as most prompt edits.
3. **Get a test-like labelled set early.** Our internal proxies could not rank strong configs. An external
   set that tracked the public LB did, and it found +0.17 F1 of fixable errors in one evening, on the last day.
4. **Validate on the exact platform stack.** vLLM version (0.19.1 vs 0.31), GPU (A100 Marlin W8A16 vs Hopper
   W8A8 FP8 numerics) and "no pip install" each changed which options were viable.
