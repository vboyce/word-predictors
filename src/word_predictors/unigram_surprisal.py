"""Unigram surprisal estimated from a causal LM's own samples: Hopton et al.
(2026)'s marginalized Monte Carlo estimator, canonical tokenization only
("MMC Approx.").

The model's distribution over words: sample strings from the model with no
prompt (after BOS) until EOS or max_tokens tokens. As in Hopton et al. (App.
C), a word starts at the start of a string or at a token that begins with a
space. Unlike Hopton et al., it ends before the next token that begins with
whitespace (a space or a newline) or with punctuation other than an
apostrophe, or before EOS: "sat." is an occurrence of "sat", "cat's" is not
one of "cat". A span starting with punctuation ('"The') is not a word. A word
counts only if the token after it is in the string, so a string cut off at
max_tokens has an incomplete last word. Words are given without outer
punctuation (see word_predictors.punctuation.strip_punctuation).

Unigram probability = expected occurrences of the word / expected number of
words. The numerator is marginalized: at every prefix of every sampled string,
add P(the word's canonical tokens, then a right-boundary token | prefix), instead
of counting the word in the sample. At the start of a string the word can
appear with or without a leading space; both canonical forms are added.
Non-canonical tokenizations of the word are not counted.

Supports tokenizers that mark a preceding space on the following token, like
GPT-2's and Pythia's byte-level BPE ("Ġcat"); anything else raises.
"""

import copy
import math
from collections import namedtuple

import torch

from word_predictors.contextual_surprisal import model_context_window
from word_predictors.punctuation import APOSTROPHES, PUNCTUATION, strip_punctuation

SPACE_MARKER = "Ġ"
# Batch sizes bound memory: scoring holds a KV cache per word in the batch and
# logits over the whole vocabulary for each token of each word.
SAMPLES_PER_BATCH = 256
WORDS_PER_BATCH = 64

# Token ids that can start a word, and that end the word before them.
WordBoundaries = namedtuple("WordBoundaries", ["word_start_ids", "right_boundary_ids"])


def unigram_surprisal(words, tokenizer, model, n_samples, max_tokens, seed):
    """Unigram surprisal (bits) of each word. Returns one dict per word, in
    order, with word, unigram_surprisal, and the sampling settings."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    window = model_context_window(model.config)
    if not 0 < max_tokens < window:
        raise ValueError(f"max_tokens must be between 1 and {window - 1} "
                         f"(BOS plus max_tokens must fit in the context window), got {max_tokens}")
    boundaries = word_boundaries(tokenizer)
    for word in words:
        canonical_forms(word, tokenizer, boundaries)  # fail before sampling, not after

    strings = sample_strings(model, tokenizer.bos_token_id, tokenizer.eos_token_id,
                             n_samples, max_tokens, seed)
    total_words = sum(complete_word_count(string, tokenizer, boundaries) for string in strings)
    occurrences = expected_occurrences(strings, words, tokenizer, model, max_tokens)
    rows = []
    for word in words:
        rows.append({
            "word": word,
            "unigram_surprisal": -math.log2(sum(occurrences[word]) / total_words),
            "n_samples": n_samples,
            "max_tokens": max_tokens,
            "seed": seed,
        })
    return rows


def canonical_forms(word, tokenizer, boundaries):
    """(tokens at the start of a string, tokens after a space) for word."""
    if word == "" or any(character.isspace() for character in word):
        raise ValueError(f"word {word!r} is empty or contains whitespace")
    if strip_punctuation(word) != word:
        raise ValueError(f"word {word!r} has outer punctuation: pass {strip_punctuation(word)!r}")
    start_form = tokenizer.encode(word, add_special_tokens=False)
    space_form = tokenizer.encode(" " + word, add_special_tokens=False)
    start_tokens = tokenizer.convert_ids_to_tokens(start_form)
    space_tokens = tokenizer.convert_ids_to_tokens(space_form)
    if start_tokens[0].startswith(SPACE_MARKER) or not space_tokens[0].startswith(SPACE_MARKER):
        raise ValueError(
            f"tokenizer {tokenizer.name_or_path!r} doesn't use the expected scheme (a space "
            f"marked as {SPACE_MARKER!r} on the next token): {word!r} -> {start_tokens}, "
            f"{' ' + word!r} -> {space_tokens}"
        )
    if any(token in boundaries.right_boundary_ids for token in start_form[1:] + space_form[1:]):
        raise ValueError(f"word {word!r} contains a word boundary: {start_tokens}, {space_tokens}")
    return start_form, space_form


def word_boundaries(tokenizer):
    """WordBoundaries for tokenizer: word starts are the tokens marked with a
    preceding space; right boundaries are EOS and the tokens whose text starts
    with whitespace or with punctuation other than an apostrophe."""
    vocab = tokenizer.get_vocab()
    word_start_ids = {i for token, i in vocab.items() if token.startswith(SPACE_MARKER)}
    right_boundary_ids = {tokenizer.eos_token_id}
    for token, i in vocab.items():
        text = tokenizer.convert_tokens_to_string([token])
        if text and (text[0].isspace() or (text[0] in PUNCTUATION and text[0] not in APOSTROPHES)):
            right_boundary_ids.add(i)
    return WordBoundaries(frozenset(word_start_ids), frozenset(right_boundary_ids))


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


def complete_word_count(string, tokenizer, boundaries):
    """Number of words in string: spans from a word start to the next right
    boundary, which must be in the string, not starting with punctuation."""
    count = 0
    for start, token in enumerate(string):
        if token == tokenizer.eos_token_id or not (start == 0 or token in boundaries.word_start_ids):
            continue
        end = next((i for i in range(start + 1, len(string))
                    if string[i] in boundaries.right_boundary_ids), None)
        if end is None:
            continue
        text = tokenizer.decode(string[start:end]).lstrip()
        if text and text[0] not in PUNCTUATION:
            count += 1
    return count


def expected_occurrences(strings, words, tokenizer, model, max_tokens):
    """{word: [for each string, the sum over its prefixes of P(word's canonical
    tokens, then a right-boundary token | prefix)]}: the expected number of
    occurrences of each word given each string's prefixes."""
    boundaries = word_boundaries(tokenizer)
    forms = {word: canonical_forms(word, tokenizer, boundaries) for word in words}
    boundary_ids = sorted(boundaries.right_boundary_ids)
    totals = {word: [0.0] * len(strings) for word in words}
    for string_index, string in enumerate(strings):
        ended = string[-1] == tokenizer.eos_token_id
        prefix_tokens = string[:-1] if ended else string
        for batch_start in range(0, len(words), WORDS_PER_BATCH):
            batch = words[batch_start:batch_start + WORDS_PER_BATCH]
            # Only at the start of the string can the word come without a leading space.
            at_start = summed_over_prefixes([], [forms[w][0] for w in batch],
                                            tokenizer, model, boundary_ids, max_tokens)
            after_space = summed_over_prefixes(prefix_tokens, [forms[w][1] for w in batch],
                                               tokenizer, model, boundary_ids, max_tokens)
            for word, start_total, space_total in zip(batch, at_start, after_space):
                totals[word][string_index] += start_total + space_total
    return totals


def summed_over_prefixes(prefix_tokens, forms, tokenizer, model, boundary_ids, max_tokens):
    """For each form, the sum over j = 0..len(prefix_tokens) of P(form, then
    one of boundary_ids | BOS + prefix_tokens[:j]), counting only the j where
    the form and its boundary token fit in max_tokens."""
    # Every row of the batch carries the same prefix, so the prefix's KV cache
    # is built once, one token at a time, and each prefix scores all forms at once.
    n_forms = len(forms)
    longest = max(len(form) for form in forms)
    # Right padding needs no attention mask: a causal LM's logits at a
    # position don't depend on later tokens.
    padded_forms = torch.tensor([form + [tokenizer.eos_token_id] * (longest - len(form))
                                 for form in forms], device=model.device)
    form_lengths = [len(form) for form in forms]
    totals = [0.0] * n_forms
    cache = None
    for prefix_length in range(len(prefix_tokens) + 1):
        fits = [prefix_length + length + 1 <= max_tokens for length in form_lengths]
        if not any(fits):
            break
        last_token = tokenizer.bos_token_id if prefix_length == 0 else prefix_tokens[prefix_length - 1]
        with torch.no_grad():
            prefix_output = model(torch.full((n_forms, 1), last_token, device=model.device),
                                  past_key_values=cache, use_cache=True)
            cache = prefix_output.past_key_values
            form_logits = model(padded_forms, past_key_values=copy.deepcopy(cache)).logits
        # Position i predicts the form's token i; position len(form) predicts the boundary.
        logits = torch.cat([prefix_output.logits[:, -1:], form_logits], dim=1)
        for row, form in enumerate(forms):
            if fits[row]:
                totals[row] += word_then_boundary_probability(logits[row], form, boundary_ids)
    return totals


def word_then_boundary_probability(logits, form, boundary_ids):
    """P(form, then one of boundary_ids), from logits whose position i predicts
    the form's token i and position len(form) the token after it."""
    log_probs = torch.log_softmax(logits[:len(form) + 1].double(), dim=-1)
    log_probability = sum(log_probs[i, token].item() for i, token in enumerate(form))
    boundary = torch.logsumexp(log_probs[len(form), boundary_ids], dim=0)
    return math.exp(log_probability + boundary.item())
