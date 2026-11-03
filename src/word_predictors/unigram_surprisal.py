"""Unigram surprisal estimated from a causal LM's own samples: Hopton et al.
(2026)'s marginalized Monte Carlo estimator, canonical tokenization only
("MMC Approx.").

The model's distribution over words: sample strings from the model with no
prompt (after BOS) until EOS or max_tokens tokens. A word starts at the start
of a string or at a token that begins with a space, and ends before the next
such token or EOS (Hopton et al. App. C), so attached punctuation is part of
the word ("sat." is not an occurrence of "sat"). A word counts only if the
token after it is in the string, so a string cut off at max_tokens has an
incomplete last word.

Unigram probability = expected occurrences of the word / expected number of
words. The numerator is marginalized: at every prefix of every sampled string,
add P(the word's canonical tokens, then a word start or EOS | prefix), instead
of counting the word in the sample. At the start of a string the word can
appear with or without a leading space; both canonical forms are added.
Non-canonical tokenizations of the word are not counted.

Supports tokenizers that mark a preceding space on the following token, like
GPT-2's and Pythia's byte-level BPE ("Ġcat"); anything else raises.
"""

import math

import torch

from word_predictors.contextual_surprisal import context_window

SPACE_MARKER = "Ġ"
# Batch sizes bound memory: scoring holds logits over the whole vocabulary at
# every position of every sequence in the batch (32 x 50 x 50k floats is 320 MB).
SAMPLES_PER_BATCH = 256
SEQUENCES_PER_BATCH = 32


def unigram_surprisal(words, tokenizer, model, n_samples, max_tokens, seed):
    """Unigram surprisal (bits) of each word. Returns one dict per word, in
    order, with word, unigram_surprisal, and the sampling settings."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    window = context_window(model.config)
    if not 0 < max_tokens < window:
        raise ValueError(f"max_tokens must be between 1 and {window - 1} "
                         f"(BOS plus max_tokens must fit in the context window), got {max_tokens}")
    for word in words:
        canonical_forms(word, tokenizer)  # fail before sampling, not after

    strings = sample_strings(model, tokenizer.bos_token_id, tokenizer.eos_token_id,
                             n_samples, max_tokens, seed)
    total_words = sum(complete_word_count(string, tokenizer) for string in strings)
    rows = []
    for word in words:
        occurrences = sum(expected_occurrences(strings, word, tokenizer, model, max_tokens))
        rows.append({
            "word": word,
            "unigram_surprisal": -math.log2(occurrences / total_words),
            "n_samples": n_samples,
            "max_tokens": max_tokens,
            "seed": seed,
        })
    return rows


def canonical_forms(word, tokenizer):
    """(tokens at the start of a string, tokens after a space) for word."""
    if word == "" or any(character.isspace() for character in word):
        raise ValueError(f"word {word!r} is empty or contains whitespace")
    start_form = tokenizer.encode(word, add_special_tokens=False)
    space_form = tokenizer.encode(" " + word, add_special_tokens=False)
    start_tokens = tokenizer.convert_ids_to_tokens(start_form)
    space_tokens = tokenizer.convert_ids_to_tokens(space_form)
    if start_tokens[0].startswith(SPACE_MARKER) or not space_tokens[0].startswith(SPACE_MARKER) \
            or any(t.startswith(SPACE_MARKER) for t in start_tokens[1:] + space_tokens[1:]):
        raise ValueError(
            f"tokenizer {tokenizer.name_or_path!r} doesn't use the expected scheme (a space "
            f"marked as {SPACE_MARKER!r} on the next token): {word!r} -> {start_tokens}, "
            f"{' ' + word!r} -> {space_tokens}"
        )
    return start_form, space_form


def word_start_ids(tokenizer):
    """Ids of the tokens that begin with a space."""
    return {i for token, i in tokenizer.get_vocab().items() if token.startswith(SPACE_MARKER)}


def sample_strings(model, bos_id, eos_id, n_samples, max_tokens, seed):
    """n_samples strings sampled from the model after BOS, each ending at its
    first EOS (included) or cut off at max_tokens tokens."""
    # Plain ancestral sampling, rather than model.generate, whose defaults
    # (e.g. a top_k from the model's generation config) would change the distribution.
    generator = torch.Generator(device=model.device).manual_seed(seed)
    strings = []
    for batch_start in range(0, n_samples, SAMPLES_PER_BATCH):
        batch_size = min(SAMPLES_PER_BATCH, n_samples - batch_start)
        ids = torch.full((batch_size, 1), bos_id, device=model.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
        cache = None
        for _ in range(max_tokens):
            with torch.no_grad():
                output = model(ids[:, -1:], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[:, -1].double()
            next_ids = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)[:, 0]
            next_ids[finished] = eos_id  # filler after a string has ended
            ids = torch.cat([ids, next_ids[:, None]], dim=1)
            finished |= next_ids == eos_id
            if finished.all():
                break
        for row in ids[:, 1:].tolist():
            strings.append(row[:row.index(eos_id) + 1] if eos_id in row else row)
    return strings


def complete_word_count(string, tokenizer):
    """Number of words in string followed by a word start or EOS."""
    starts = word_start_ids(tokenizer)
    ended = string[-1] == tokenizer.eos_token_id
    tokens = string[:-1] if ended else string
    n_word_starts = sum(1 for i, token in enumerate(tokens) if i == 0 or token in starts)
    return n_word_starts if ended else max(n_word_starts - 1, 0)


def expected_occurrences(strings, word, tokenizer, model, max_tokens):
    """For each string, the sum over its prefixes of P(word's canonical tokens,
    then a word start or EOS | prefix): the expected number of occurrences of
    word given the string's prefixes."""
    start_form, space_form = canonical_forms(word, tokenizer)
    boundary_ids = sorted(word_start_ids(tokenizer) | {tokenizer.eos_token_id})
    sequences, owners = [], []
    for string_index, string in enumerate(strings):
        prefix_limit = len(string) - 1 if string[-1] == tokenizer.eos_token_id else len(string)
        for prefix_length in range(prefix_limit + 1):
            forms = [start_form, space_form] if prefix_length == 0 else [space_form]
            for form in forms:
                if prefix_length + len(form) + 1 <= max_tokens:  # the boundary token must fit
                    sequences.append(([tokenizer.bos_token_id] + string[:prefix_length], form))
                    owners.append(string_index)
    totals = [0.0] * len(strings)
    for batch_start in range(0, len(sequences), SEQUENCES_PER_BATCH):
        batch = sequences[batch_start:batch_start + SEQUENCES_PER_BATCH]
        probabilities = word_then_boundary_probabilities(batch, model, boundary_ids,
                                                         tokenizer.eos_token_id)
        for owner, probability in zip(owners[batch_start:], probabilities):
            totals[owner] += probability
    return totals


def word_then_boundary_probabilities(batch, model, boundary_ids, pad_id):
    """P(form, then one of boundary_ids | context) for each (context, form) in batch."""
    full = [context + form for context, form in batch]
    length = max(len(ids) for ids in full)
    # Right padding needs no attention mask: a causal LM's logits at a
    # position don't depend on later tokens.
    padded = torch.tensor([ids + [pad_id] * (length - len(ids)) for ids in full], device=model.device)
    with torch.no_grad():
        logits = model(padded).logits
    results = []
    for row, (context, form) in enumerate(batch):
        # The positions whose next-token distributions predict the word's tokens, then the boundary.
        first = len(context) - 1
        log_probs = torch.log_softmax(logits[row, first:first + len(form) + 1].double(), dim=-1)
        log_probability = sum(log_probs[i, token].item() for i, token in enumerate(form))
        boundary = torch.logsumexp(log_probs[len(form), boundary_ids], dim=0)
        results.append(math.exp(log_probability + boundary.item()))
    return results
