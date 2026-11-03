# word-predictors: design (draft, 2026-11-03)

Background and sources: `~/notes/claude/Kid-maze LM and frequency methods.md`.

## What it computes

Input: stories as words in order (columns `topic, sentence, word_position, word`, as in
`kid-maze-passages/data_clean/passages.csv`). Output: one row per word, per model or corpus.

1. **Contextual surprisal** (bits) from a Hugging Face causal LM, with the story so far as
   context. Summed over the word's tokens, plus `n_tokens`. Same definition as the current
   `kid-maze-passages/analysis/word_predictors.py`, generalized to any causal LM. The context
   window comes from the model config; no default.
2. **Unigram surprisal from the same LM** (bits): Hopton et al. 2026's marginalized Monte Carlo
   estimator, canonical-tokenization-only ("MMC Approx."):
   - sample N strings of at most L tokens from the model with no prompt;
   - at every sampled prefix, add P(word's canonical tokens, then a word boundary | prefix);
   - divide by the expected number of words in the samples.
   N, L and the seed are required arguments and are written to the output.
3. **N-gram surprisal** (bits), n = 1..5, from corpus counts with Stupid Backoff (α = 0.4), as
   in Michaelov & Levy 2026:
   - the Pile via the infini-gram API (paired with Pythia);
   - local counts for the small developmental corpora (BabyLM, CHILDES, TinyStories), paired
     with the models trained on them.
4. `pmi = unigram_surprisal − surprisal`, computed downstream in the study repo, not here.

## Models (first round)

- Pythia, several sizes (70m … 12b; deduped or standard, to decide), maybe training checkpoints.
  Training corpus: the Pile.
- BabyLM GPT-2 baselines, 10M and 100M words (2025 and/or 2026 corpus). Training corpus: BabyLM.
- `mcxfrank/childes-gpt2-ladder`, top budget (24M words). Training corpus: CHILDES.
- A TinyStories model, e.g. `roneneldan/TinyStories-33M`. Training corpus: TinyStories.

Tokenizer schemes (checked 2026-11-03, from the local HF cache; the Hub was unreachable):
`gpt2`, `EleutherAI/pythia-70m` and `roneneldan/TinyStories-33M` all use byte-level BPE with
Ġ-marked spaces, BOS = `<|endoftext|>`, and add no special tokens on `encode`. Context window:
GPT-2 1024 (`n_positions`), Pythia and TinyStories 2048 (`max_position_embeddings`). Not checked
yet: `mcxfrank/childes-gpt2-ladder` (not cached; its tokenizer didn't load from the repo root,
maybe it's in a per-budget subfolder) and the BabyLM baselines (repo names not chosen).
Word-boundary detection for the unigram estimator must handle whichever schemes these use, and
fail loudly on a tokenizer it doesn't recognize.

## Unigram surprisal (implemented; real runs need a GPU)

`word_predictors.unigram_surprisal.unigram_surprisal(words, tokenizer, model, n_samples,
max_tokens, seed)`. Definitions as built (check these):
- A word starts at the start of a string or at a Ġ-token, and ends before the next Ġ-token or
  EOS (Hopton App. C). So **attached punctuation is part of the word**: "sat." is not an
  occurrence of "sat". Whether to pass words with or without punctuation, or to make punctuation
  a right boundary (as planned for n-grams), is **still open**.
- At the start of a sampled string the word may appear with or without a leading space; both
  canonical forms are counted.
- Strings cut off at `max_tokens` have an incomplete last word: it counts in neither the
  numerator nor the denominator. The word plus its boundary token must fit in `max_tokens`.
- Denominator: the sample mean of complete words per string (not marginalized).
- Sampling is plain ancestral sampling, not `model.generate` (whose generation-config defaults,
  e.g. top_k, would change the distribution).
- Only tokenizers with Ġ-marked spaces; anything else raises.

Tests: exact unbiasedness of the marginalized count against enumerating every string from a
7-token model, and convergence to the exact surprisal (error < 1 bootstrap SE at N = 2000 and
20000).

**Cost.** Each prefix's KV cache is built once per string and scores a batch of words at once
(2× faster than one pass per (sample, prefix, word): GPT-2, 4 samples × 12 tokens × 20 words,
22.9 s → 12.0 s on 2 CPU threads, same values to 2e-5 bits). What remains is intrinsic: a
full-vocabulary distribution at every (prefix, word, word token), about N × L × W × (k + 1)
rows of the output layer. Rough estimate for GPT-2 at N = 500, L = 50, W = 400: ~1e15
multiply-adds, i.e. days on a laptop CPU, minutes to tens of minutes on a GPU (not measured).
Run on Engaging.

## Contextual surprisal (implemented)

`word_predictors.contextual_surprisal.contextual_surprisal(word_rows, tokenizer, model,
window_stride)`, with `load_causal_lm(model_name)`. `window_stride` is required; the context
window comes from the model config (`n_positions` or `max_position_embeddings`; error if
neither, or if they disagree). Errors on: no BOS token, word-by-word tokenization that differs
from whole-story tokenization, empty words or words containing whitespace, duplicate
(sentence, word_position). Surprisal is not rounded (kid-maze rounds to 4 decimals on writing).
With GPT-2 and `window_stride=512` it reproduces kid-maze's `word_predictors.csv` (all 606 words
within 1e-4 bits; fixture in `tests/data/`).

## Word boundaries

A word counts only if it starts at a word boundary and is followed by one (Hopton App. C).
Michaelov & Levy's code does not enforce the right boundary: it sends text to the infini-gram
API, which counts token sequences, so "cat" also counts the start of "category". Options for
the Pile: use infini-gram's next-token-distribution query to get the share of continuations
that start a new word (to check: whether that query is exact or sampled for frequent prefixes).
Local corpora: count whole words directly.

Attached punctuation: stripped for n-gram counts (like `length` and `zipf_frequency` now);
kept for contextual surprisal (as the LM sees it). Punctuation counts as a right boundary.

## Fail loudly (differences from Michaelov & Levy's code)

- No bare `except`/retry loops. An API error stops the run with the query that failed.
- A word with zero corpus count: their code silently uses count 1. Here: decision needed (see
  below); whichever it is, the output records the raw count so it's visible.

## Tests (plan)

- A tiny randomly initialized causal LM built in the test (like
  `maze-distractor-generator/tests/tiny_lm.py`) with a small vocabulary.
- Contextual surprisal: matches a brute-force token-by-token computation, and the existing
  `word_predictors.csv` values for one real model (integration test, needs the model download;
  it fails, not skips, without it).
- Unigram estimator: with a tiny vocabulary and short L, the expected word count can be computed
  exactly by enumerating all strings; the estimate must converge to it as N grows, and the
  canonical-only version must equal it when the tiny tokenizer has one tokenization per word.
- N-grams: hand-countable toy corpus; Stupid Backoff values computed by hand; API client tested
  against recorded responses, plus one explicitly-run live test.

## Decisions (2026-11-03)

1. **infini-gram API: yes**, the passage text can be sent (the stories are already public).
2. **N-gram context does not cross sentence boundaries.** The first word of a sentence gets
   only its unigram; the second word at most a bigram, and so on. (Contextual LM surprisal
   still uses the story so far.)

## Still open

1. **Zero counts.** Recommendation: don't pick a rule yet. The first n-gram run writes raw
   counts for every order; count the zero unigrams per corpus, then decide. Higher-order zeros
   are already handled by backoff. Only a zero *unigram* needs a rule, and in the Pile it
   probably won't happen for these words; in CHILDES (24M words) words like "Tyrannosaurus"
   may be unseen. If only a few, leaving them undefined (dropped from n-gram fits) is cleanest;
   Michaelov & Levy's count-1 rule puts every unseen word at the same floor value.
2. **Pythia.** Recommendation:
   - standard (not deduped): the infini-gram Pile index is the standard Pile train set, so
     standard Pythia is the one whose training data the counts match;
   - sizes 14m through 2.8b first (cheap; Shain 2024 and de Varda 2026 find small models fit
     behavior best), 6.9b and 12b only if the trend calls for it;
   - training checkpoints later, as a separate question (training tokens as a stand-in for
     amount of experience, as in Michaelov & Levy).
3. Multi-token words: report summed surprisal (lab default) and also fit with them excluded
   (Boyce & Levy 2023)? That's a study-repo choice; this library just reports `n_tokens`.
