"""Contextual surprisal from a Hugging Face causal LM.

Each story is read in order, so earlier sentences are context. A word's
surprisal (bits) is summed over its tokens; it includes attached punctuation,
as the model sees it. Each word is tokenized with its preceding space, and the
story starts after the tokenizer's BOS token. A story longer than the model's
context window is scored in windows of that many tokens, one starting every
window_stride tokens; each token is scored in the first window that contains
it, so it has at least (context window - window_stride) tokens of context.
"""

import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config attributes that give the maximum sequence length, depending on the architecture.
CONTEXT_WINDOW_ATTRIBUTES = ["n_positions", "max_position_embeddings"]


def load_causal_lm(model_name, subfolder=""):
    """Return (tokenizer, model) for a Hugging Face causal LM, in eval mode and float32.
    subfolder is for repos holding several models, e.g.
    mcxfrank/childes-gpt2-ladder's "development/seed42/rung24M"."""
    # For a missing subfolder, transformers raises misleading errors ("couldn't
    # instantiate the backend tokenizer", "unrecognized model"); local paths can be checked first.
    local_folder = Path(model_name) / subfolder
    if Path(model_name).is_dir() and not local_folder.is_dir():
        raise FileNotFoundError(f"no model folder {local_folder}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
    # float32 whatever the stored precision: transformers loads a model in the
    # dtype it was saved in (Pythia: fp16), and fp16 results on CPU differ
    # between processors by up to 1.5 bits per word.
    model = AutoModelForCausalLM.from_pretrained(model_name, subfolder=subfolder, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def model_context_window(config):
    """The maximum number of tokens the model can take, from its config."""
    values = {
        attribute: getattr(config, attribute)
        for attribute in CONTEXT_WINDOW_ATTRIBUTES
        if getattr(config, attribute, None) is not None
    }
    if len(set(values.values())) != 1:
        raise ValueError(
            f"can't tell the context window from the model config: expected one of "
            f"{CONTEXT_WINDOW_ATTRIBUTES}, got {values or 'none'}"
        )
    return next(iter(values.values()))


def contextual_surprisal(word_rows, tokenizer, model, window_stride, context_window=None):
    """Surprisal for every word in word_rows (dicts with topic, sentence,
    word_position, word). Returns one dict per word, stories in order of first
    appearance and words in reading order, with topic, sentence and
    word_position (as int), word, surprisal (bits) and n_tokens.

    context_window: the longest sequence to score, if the model was trained on
    shorter sequences than its config allows (TinyStories-33M: 512 of 2048);
    otherwise the config's."""
    if tokenizer.bos_token_id is None:
        raise ValueError(f"tokenizer {tokenizer.name_or_path!r} has no BOS token to start a story with")
    window = model_context_window(model.config)
    if context_window is not None:
        if context_window > window:
            raise ValueError(f"context_window {context_window} is longer than the model's {window}")
        window = context_window
    if not 0 < window_stride < window:
        raise ValueError(f"window_stride must be between 1 and {window - 1}, got {window_stride}")

    results = []
    for topic, entries in group_stories(word_rows).items():
        words = [word for _, _, word in entries]
        surprisals = story_surprisals(words, tokenizer, model, window, window_stride)
        for (sentence, word_position, word), (surprisal, n_tokens) in zip(entries, surprisals):
            results.append({
                "topic": topic,
                "sentence": sentence,
                "word_position": word_position,
                "word": word,
                "surprisal": surprisal,
                "n_tokens": n_tokens,
            })
    return results


def group_stories(word_rows):
    """Return {topic: [(sentence, word_position, word), ...]} in reading order."""
    stories = defaultdict(list)
    for row in word_rows:
        word = row["word"]
        if word == "" or any(character.isspace() for character in word):
            raise ValueError(f"{row['topic']} {row['sentence']}.{row['word_position']}: "
                             f"word {word!r} is empty or contains whitespace")
        stories[row["topic"]].append((int(row["sentence"]), int(row["word_position"]), word))
    for topic, entries in stories.items():
        entries.sort()
        positions = [(sentence, word_position) for sentence, word_position, _ in entries]
        if len(set(positions)) != len(positions):
            raise ValueError(f"{topic}: duplicate (sentence, word_position)")
    return dict(stories)


def story_surprisals(words, tokenizer, model, window, window_stride):
    """(surprisal in bits, n_tokens) for each word, with the story so far as context."""
    word_tokens = [
        tokenizer.encode(word if i == 0 else " " + word, add_special_tokens=False)
        for i, word in enumerate(words)
    ]
    token_ids = [tokenizer.bos_token_id] + [t for tokens in word_tokens for t in tokens]
    whole_text = tokenizer.encode(" ".join(words), add_special_tokens=False)
    if token_ids[1:] != whole_text:
        raise ValueError(
            f"tokenizer {tokenizer.name_or_path!r}: tokenizing word by word differs from "
            f"tokenizing the whole story (story starting {' '.join(words[:5])!r})"
        )

    log_probs = token_log_probs(token_ids, model, window, window_stride)

    results = []
    position = 1  # index in token_ids of the current word's first token
    for tokens in word_tokens:
        surprisal = -sum(log_probs[position:position + len(tokens)]) / math.log(2)
        results.append((surprisal, len(tokens)))
        position += len(tokens)
    return results


def token_log_probs(token_ids, model, window, window_stride):
    """Natural-log probability of each token given the tokens before it (None
    for the first token). Windows start every window_stride tokens; each scores
    only the tokens the previous window didn't."""
    n = len(token_ids)
    log_probs = [None] * n
    start = 0
    while True:
        end = min(start + window, n)
        with torch.no_grad():
            logits = model(torch.tensor([token_ids[start:end]])).logits[0]
        window_log_probs = torch.log_softmax(logits, dim=-1)
        first_unscored = 1 if start == 0 else start - window_stride + window
        for i in range(first_unscored, end):
            log_probs[i] = window_log_probs[i - start - 1, token_ids[i]].item()
        if end == n:
            break
        start += window_stride
    return log_probs
