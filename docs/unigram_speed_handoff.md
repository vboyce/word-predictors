# Handoff: speed up `unigram_surprisal` (written 2026-11-05)

Written from a kid-maze-passages session for the next session in this repo.
`src/word_predictors/` is reusable code: TDD rules apply (`~/.claude/project-types/reusable.md`).
Check whether prove_it is on for this repo before starting.

## Problem

`kid-maze-passages/analysis/pythia_unigram_surprisal.py` calls
`word_predictors.unigram_surprisal.unigram_surprisal` for the 325 distinct expt-1 story words.
For **pythia-70m, 100 samples of up to 256 tokens**, on Engaging:

| job | GPU | seed | wall time |
|---|---|---|---|
| 27914034 | H100 80GB | 1 | 20,279 s (5.6 h) |
| 27914035 | A100 80GB | 2 | 31,400 s (8.7 h) |

Outputs: `kid-maze-passages/analysis/models/pythia_unigram/70m_n100_L256_seed{1,2}.csv`.
The kid-maze predictor-source models need 410m and 1.4b too, and probably more samples (below),
so this has to get much faster first.

**Seed-to-seed spread at 100 samples** (the two CSVs above, all 325 words): r = 0.993; |difference|
median 0.21 bits, 90th percentile 0.73, max 2.05 (e.g. "species" 12.4 vs. 14.4). That's a lot of
noise for a predictor, so expect to want 1,000+ samples, which at today's speed is 2–4 days for 70m.

## Diagnosis (from reading the code, NOT profiled)

All in `src/word_predictors/unigram_surprisal.py`. Sampling (`sample_strings`, batched, KV-cached)
looks cheap. The time is almost certainly in `expected_occurrences` → `summed_over_prefixes`:

1. **Loop structure.** For each of 100 strings × 6 word batches (`WORDS_PER_BATCH = 64`, line 40)
   × up to 256 prefix positions, `summed_over_prefixes` (line 178) runs two forward passes: one
   on the prefix's next token (64 identical rows), one on the padded forms.
2. **`copy.deepcopy(cache)` per prefix position** (line 202): a deep copy of a KV cache that
   grows with the prefix and is replicated 64 times (one row per word).
3. **The prefix's KV cache is rebuilt from scratch for every word batch**, 6 times per string,
   and with 64 identical rows each time.
4. **`word_then_boundary_probability`** (line 211) is called in Python for every word at every
   prefix: about 100 × 6 × 256 × 64 ≈ 10 million calls. Each does a float64 `log_softmax` over
   the whole vocabulary (~50k) at len(form) + 1 positions, plus `.item()` GPU syncs (line 215, 217).
   This is my best guess for the dominant cost.
5. **`at_start`** (line 169) doesn't depend on the string but is recomputed for each of the 100
   strings.

Profile before rewriting (e.g. `torch.profiler` or simple timers around the forward passes, the
deepcopy, and the scoring loop, on 70m with a few strings on CPU or GPU) to confirm the ranking.

## Fix ideas, roughly in order of payoff

- Score on the GPU in one go: one `log_softmax` per batch, then gather the form tokens and
  `logsumexp` over the boundary ids for all words at once; no per-word Python loop, no `.item()`
  until the end (accumulate totals as a tensor).
- Compute `at_start` once, not per string.
- Run each string once (batch size 1) to get logits at every prefix: that gives P(first token of
  every form | prefix_j) for all j at once. Only the continuation (later form tokens + the
  boundary) needs extra passes.
- Batch the continuation passes across prefixes rather than (or as well as) across words, e.g.
  expand the string's cache per prefix, or use a 4D attention mask so all (prefix, form)
  continuations go in one pass. Check what GPT-NeoX in the installed transformers (5.19.0 on
  Engaging) supports before choosing.
- At minimum, if the structure stays: build the prefix cache once per string (not per word batch)
  and avoid `deepcopy` (e.g. a cache that can be cropped back after the form pass).

## Goals / non-goals / success criteria

- Goal: same estimator (canonical tokenization, marginalized counts, same word-boundary rules),
  much faster.
- Non-goal: changing the estimator, boundary rules, or sampling. Sampling should stay
  bit-for-bit the same given a seed and device.
- Success:
  - The existing tests in `tests/test_unigram_surprisal.py` pass unchanged (unbiasedness,
    convergence to the exact value on the enumerable tiny LM, etc.).
  - A new test: the new `expected_occurrences` matches the current one on fixed strings with the
    tiny LM, to floating-point tolerance. Keep the current implementation as a reference in
    the test (not in `src/`) for that.
  - A timing on 70m (e.g. 10 strings) before and after, reported with the GPU type.
  - Real-data check: rerun 70m seed 1 on an **H100** (sampling can differ across GPU types) and
    compare with `70m_n100_L256_seed1.csv` (up to float tolerance).

## Open questions for V

- How many samples to target (see the seed spread above)? Picking it needs the new timing.
- 410m and 1.4b: same samples count, or fewer for the bigger models?
