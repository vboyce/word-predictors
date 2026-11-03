# Local corpora

Training corpora of the LMs, for whole-word n-gram counts
(`word_predictors.ngram_surprisal.count_ngrams`). Files are not in git; each
subfolder is listed here with its source, version, and download date.

| Subfolder | Corpus | Source | Downloaded | Paired model |
|---|---|---|---|---|
| `tinystories/` | TinyStories (original, GPT-3.5 + GPT-4), `TinyStories-train.txt` (2,119,718 stories, 373.8M whitespace words) and `-valid.txt` | HF `roneneldan/TinyStories` @ `f54c09fd`, sha256-checked | 2026-11-03 | `roneneldan/TinyStories-33M` (the dataset card says the paper's models used `TinyStories-train.txt`) |
| `childes/` | English CHILDES, Feng et al. (2024) preparation, `CHILDES_data.zip` unzipped (train/val; ordered, randomized, reversed; with and without speaker labels) | GitHub `styfeng/TinyDialogues` @ `1ba0423a`, `data/CHILDES_data.zip`, sha256-checked | 2026-11-03 | `mcxfrank/childes-gpt2-ladder` (model card: "Feng et al. 2024 preparation"; each seed saw a shuffled prefix) |

Reading them: `word_predictors.corpora.childes_utterances` (use `CHILDES_train_ordered.txt`:
4,169,545 utterances, 16,127,826 words without labels) and `tinystories_lines`.
CHILDES data is subject to TalkBank's terms of use.
