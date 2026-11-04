# word-predictors: design (draft, 2026-11-03)

Background and sources: `~/notes/claude/Kid-maze LM and frequency methods.md`.

## What it computes

Input: stories as words in order (columns `topic, sentence, word_position, word`, as in
`kid-maze-passages/data_clean/passages.csv`). Output: one row per word, per model or corpus.

1. **Contextual surprisal** (bits) from a Hugging Face causal LM, with the story so far as
   context. Summed over the word's tokens, plus `n_tokens`. Same definition as the current
   `kid-maze-passages/analysis/gpt2_word_predictors.py`, generalized to any causal LM. The context
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
GPT-2 1024 (`n_positions`), Pythia and TinyStories 2048 (`max_position_embeddings`). **But
TinyStories-33M was trained with `context_length=512`** (its model card), so positions past 512
are untrained: pass `contextual_surprisal(..., context_window=512)`. The config can't show
this; other models' training lengths aren't checked.
`mcxfrank/childes-gpt2-ladder` holds 180 models in subfolders `development/seed{S}/rung{B}/`
(10 seeds × 18 budgets, 0.5M–24M words; load with `load_causal_lm(name, subfolder=...)`); the
24M seed-42 tokenizer is the same Ġ scheme, BOS = EOS = `<|endoftext|>` (id 0), 52k vocab,
context 1024. Which seed(s) to use is open. Not checked yet: the BabyLM baselines (repo names
not chosen).
Word-boundary detection for the unigram estimator must handle whichever schemes these use, and
fail loudly on a tokenizer it doesn't recognize.

## Unigram surprisal (implemented; real runs need a GPU)

`word_predictors.unigram_surprisal.unigram_surprisal(words, tokenizer, model, n_samples,
max_tokens, seed)`. Definitions as built (check these):
- A word starts at the start of a string or at a Ġ-token (Hopton App. C). It ends before EOS
  or a token whose text starts with whitespace (space, newline) or punctuation other than an
  apostrophe (decided 2026-11-03; Hopton's rule is Ġ-token or EOS only, and the paper doesn't
  discuss punctuation). So "sat." is an occurrence of "sat"; "cat's" is not one of "cat"; a
  line-final word counts. A span starting with punctuation ('"The') is not a word, in the
  numerator or the denominator. Words are passed without outer punctuation (an error
  otherwise); a word whose tokens contain a boundary ("e.g") raises.
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

## N-gram surprisal from local counts (implemented; infini-gram not yet)

`word_predictors.ngram_surprisal`: `needed_ngrams(word_rows, max_order)` lists the n-grams the
stories need; `count_ngrams(lines, ngrams)` counts only those in a corpus (one pass; a full
5-gram table for a 100M-word corpus wouldn't fit in memory); `ngram_surprisal(word_rows,
counts, total_words, max_order, alpha)` gives one row per word and order, with the n-gram
used, `ngram_count`, `context_count`, `unigram_count`, `total_words`, `alpha`. `alpha` is
required. Choices made in code (check these):
- Counts are **case-sensitive** (as infini-gram and the LMs are): sentence-initial "The" and
  "the" are different words.
- Outer punctuation stripped with the same characters as kid-maze's `strip_punctuation`
  (hyphens kept); corpus tokens that are only punctuation are dropped, so "the ... cat"
  counts as the bigram "the cat".
- Corpus n-grams don't cross line boundaries (the corpora's sentence splitting not checked yet).
- A word with unigram count 0 gets no surprisal (None), with its zero counts in the output;
  the zero-count rule stays open (below).

## infini-gram (client implemented: option 2)

From the API docs (infini-gram.readthedocs.io/en/latest/api.html) and a few live queries:
- The Pile-train index is `v4_piletrain_llama`: 383.3B tokens, tokenized with **Llama-2's**
  SentencePiece tokenizer (▁ marks a space), not Pythia's.
- `count` of an n-gram is exact. `ntd` (next-token distribution) is **sampled** from 1000
  occurrences when the context occurs more than 1000 times (`approx: true`); `cont_cnt` is then
  scaled, in steps of `prompt_cnt / 1000`.
- Errors come back as JSON with an `error` key. No rate limit is stated.
- **Right boundary matters a lot.** `count("cat")` = 9,434,100 counts the token `▁cat`, but
  its top continuations are `ast`, `ar`, `abol` (catastrophe, catalog, catabolism), so most of
  those aren't the word "cat". For "the cat sat" (532 occurrences, exact `ntd`) nearly all
  continuations are word starts.
- **Left boundary matters too.** After a newline, a word has no ▁: `<0x0A> The` occurs
  298,141,899 times vs. `<0x0A> ▁The` 4,013,977, and `▁The` 770,633,186 overall. A text query
  "The" matches only `▁The`, so it misses ~28% of "The" (most line-initial words). Michaelov &
  Levy's text queries have both problems.

Options for whole-word counts (**decided 2026-11-03: option 2**; first words of sentences
don't matter for the analysis, so the missed line-initial forms are acceptable):
lowercase words lose little: after a newline, "the" occurs 41,559,108 times vs. 9,229,350,572
after a space (0.45%); "dinosaurs" 3,544 vs. 440,540 (0.8%) (token-sequence counts, 2026-11-03).
1. As Michaelov & Levy: plain text queries; both boundary problems ignored.
2. Right boundary from `ntd`: count × share of continuations that start a new word
   (▁-token, punctuation, newline/whitespace byte, or document end). Exact when the n-gram
   occurs ≤ 1000 times, sampled otherwise (standard error of the share ≤ 1.6 points).
   Left boundary still ▁-only.
3. As 2, plus the no-▁ form after a newline (and other whitespace/quote tokens): more
   queries per n-gram, and the left-context rule has to be spelled out.
Each needed n-gram is one `ntd` query (`prompt_cnt` is its exact count): ~3000 queries for the
two kid-maze stories at n ≤ 5.

As built (`word_predictors.infini_gram.infini_gram_counts(ngrams, index)`): whole-word count =
`prompt_cnt` minus the continuations that stay inside the word. A continuation starts a new
word if its token starts with ▁, whitespace (character or byte token), or punctuation other
than an apostrophe, or is `</s>`: the unigram estimator's right boundary. Byte tokens ≥ 0x80
(pieces of curly quotes, dashes, non-ASCII letters) are not boundaries. Example (recorded,
exact): "the cat sat" occurs 532 times; the one continuation inside a word is `';` (a closing
quote, missed by the apostrophe rule), so the count is 531. Returns the counts and, per
n-gram, the token-sequence count and the `approx` flag. Any API error raises with the query; a
query whose first token lacks ▁ raises. No retries, no cache of responses yet.

Boundary rules compared: the unigram estimator and infini-gram need a space before the word
(left) and whitespace/punctuation after it (right). Local corpus counts strip outer punctuation
on both sides, so they also count '"The' and line-initial words. Not yet made identical.

**Total words for the Pile (estimate; used, decided 2026-11-04).** Stupid Backoff's unigram
level is count(w) / total words, and the index reports total *tokens* (383.3B), not words.
Michaelov & Levy divide by total tokens. Using tokens adds a constant to unigram-level scores,
but only for n-grams that back off to the unigram, so it isn't a harmless offset.
- No published word count: the Pile paper (arXiv 2101.00027) gives bytes and GPT-2 tokens per
  byte (0.29335); its datasheet (arXiv 2201.07311) gives document counts.
- Estimate from the empty-prompt `ntd` (a fixed, weighted sample of 1000 tokens; the same sample
  every time): word-start tokens (▁ then a letter or digit) are 45.6% of tokens (SE ≈ 1.6
  points) → **≈ 175B words** (95% CI ≈ 163–187B); counting every bare ▁ (often before digits)
  as a word start too gives 47.2% → 181B. Tokens instead of words would add log2(1/0.456) =
  1.13 bits to every unigram-level score.
- Exact count: `infini_gram.total_word_count` sums exact counts of all 30,197 word-start terms
  of the Llama-2 vocabulary (16,054 ▁-letter/digit tokens, 14,143 bare-▁ pairs). The API allows
  4 terms per OR clause and 500 tokens per query: ~7,550 queries. Run unpaced on 2026-11-03,
  it got **HTTP 403 Forbidden** partway (likely rate limiting); not re-run. At 1 query/s it
  would take ~2 hours. OR counts are flagged `approx: true`, though a 2-term OR equalled the sum
  of the exact single counts (`▁the` + `▁cat`).

## Contextual surprisal (implemented)

`word_predictors.contextual_surprisal.contextual_surprisal(word_rows, tokenizer, model,
window_stride)`, with `load_causal_lm(model_name, subfolder="")` (`subfolder` for repos holding several models). `window_stride` is required; the context
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
that start a new word (sampled for frequent prefixes; see "infini-gram" above for the options).
Local corpora: count whole words directly.

Attached punctuation: stripped for n-gram counts (like `length` and `zipf_frequency` now);
kept for contextual surprisal (as the LM sees it). Punctuation counts as a right boundary.

## Michaelov & Levy (2026), as their code does it

From `code/get_ngrams.py` and `stimuli/luke_2018_dl.stims` in their OSF project
(https://osf.io/msuwk; read 2026-11-03). The paper itself only says "raw counts of word
sequences in the corpus as calculated by infini-gram" with Stupid Backoff, following
Michaelov et al. (2025).
- Counts: infini-gram `count` of the text, so token sequences: no right boundary ("cat" counts
  "catastrophe"); left boundary only through the ▁ the tokenizer adds (line-initial missed).
- Unigram denominator: total **tokens** (`count` of the empty query).
- Zero unigram count: replaced by 1.
- Punctuation: kept. The stimuli are the text up to each word, punctuation attached ("...on
  your *phone.*"), so a target "phone." is queried as `▁phone .`.
- Context: the last n−1 words of the whole text so far, so it crosses sentence boundaries;
  the first word of the text gets only a unigram.
- α = 0.4; orders 1–5; 4 count queries per word and order (n-gram and context), no caching.
- Errors: bare `except` with up to 1000 retries; on final failure it prints a message and
  skips the row, so a word can silently be missing from the output.

Differences here: whole-word counts (right boundary), total words not tokens (open),
punctuation stripped, context within sentences, zero counts left visible (open), no retries.

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

## Decisions (2026-11-04)

3. **Pythia first**: standard (not deduped, since the infini-gram index is the standard Pile),
   sizes 14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, final checkpoint (main branch) only.
   CHILDES seeds postponed.
4. **Pile unigram denominator: the ≈175B-word estimate.** Only relative frequency matters.
5. Stupid Backoff α = 0.4, orders 1–5, as Michaelov & Levy.
6. infini-gram runs wrap `post_json` in `paced_post` (1 query/s) and `cached_post` (JSON lines
   of successful responses; errors aren't cached, so a rerun retries them and resumes).

Callers (kid-maze-passages): `analysis/pythia_surprisal.py` (contextual surprisal, all sizes;
`analysis/cluster/pythia_surprisal.slurm` on Engaging) and `analysis/pile_ngram_surprisal.py`.
(kid-maze's GPT-2 script was renamed `analysis/gpt2_word_predictors.py` so it no longer shadows
the package.)
transformers 5 configs have no `_commit_hash`; the Pythia caller loads from
`snapshot_download(...)`, whose folder name is the commit.

## Still open

1. **Zero counts.** Recommendation: don't pick a rule yet. The first n-gram run writes raw
   counts for every order; count the zero unigrams per corpus, then decide. Higher-order zeros
   are already handled by backoff. Only a zero *unigram* needs a rule, and in the Pile it
   probably won't happen for these words; in CHILDES (24M words) words like "Tyrannosaurus"
   may be unseen. If only a few, leaving them undefined (dropped from n-gram fits) is cleanest;
   Michaelov & Levy's count-1 rule puts every unseen word at the same floor value.
2. **Pythia**: decided 2026-11-04 (above). Still open: 6.9b and 12b if the trend calls for it;
   training checkpoints (training tokens as a stand-in for amount of experience, as in
   Michaelov & Levy).
3. Multi-token words: report summed surprisal (lab default) and also fit with them excluded
   (Boyce & Levy 2023)? That's a study-repo choice; this library just reports `n_tokens`.
