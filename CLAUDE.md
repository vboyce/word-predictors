# word-predictors

Word-level predictors of reading time for psycholinguistic studies, from
causal language models and corpus counts: contextual surprisal, unigram
surprisal estimated from the same LM (Hopton et al. 2026), and n-gram
surprisal from the LM's training corpus (Michaelov & Levy 2026). A library
that study repos call (first caller: `kid-maze-passages/analysis/word_predictors.py`).

Design and open decisions: `docs/design.md`.

## Folder types

Type definitions are in `~/.claude/project-types/`. Each folder below has a
CLAUDE.md that imports its type. The type is set by the file being edited, not
by the task.

| Folder | Type |
|---|---|
| `src/word_predictors/` | reusable |
| `tests/` | reusable |
| `docs/` | notes |
| `script/` | test runners (`script/test`); edit only when asked |

## Running things

- Environment: `.venv/` (Python 3.12.3, the same as kid-maze-passages). On
  Engaging, `~/word-predictors/.venv` built from
  `/orcd/pool/005/vboyce/envs/python312`, with `HF_HOME=/orcd/pool/005/vboyce/huggingface`.
- Install: `.venv/bin/pip install -e ".[test]"` (torch is the CPU build:
  install it first from `--index-url https://download.pytorch.org/whl/cpu`).
- Tests: `./script/test` (all, including `slow` tests that load real models
  from Hugging Face), `./script/test_fast` (unit tests only).
