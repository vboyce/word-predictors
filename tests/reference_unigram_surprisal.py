"""The original (slow) expected_occurrences, from commit d1bc283, kept as a
reference: the fast version in word_predictors.unigram_surprisal must give the
same values. Not used outside the tests."""

import copy
import math

import torch

from word_predictors.unigram_surprisal import canonical_forms, word_boundaries

WORDS_PER_BATCH = 64


def reference_expected_occurrences(strings, words, tokenizer, model, max_tokens):
    boundaries = word_boundaries(tokenizer)
    forms = {word: canonical_forms(word, tokenizer, boundaries) for word in words}
    boundary_ids = sorted(boundaries.right_boundary_ids)
    totals = {word: [0.0] * len(strings) for word in words}
    for string_index, string in enumerate(strings):
        ended = string[-1] == tokenizer.eos_token_id
        prefix_tokens = string[:-1] if ended else string
        for batch_start in range(0, len(words), WORDS_PER_BATCH):
            batch = words[batch_start:batch_start + WORDS_PER_BATCH]
            at_start = summed_over_prefixes([], [forms[w][0] for w in batch],
                                            tokenizer, model, boundary_ids, max_tokens)
            after_space = summed_over_prefixes(prefix_tokens, [forms[w][1] for w in batch],
                                               tokenizer, model, boundary_ids, max_tokens)
            for word, start_total, space_total in zip(batch, at_start, after_space):
                totals[word][string_index] += start_total + space_total
    return totals


def summed_over_prefixes(prefix_tokens, forms, tokenizer, model, boundary_ids, max_tokens):
    n_forms = len(forms)
    longest = max(len(form) for form in forms)
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
        logits = torch.cat([prefix_output.logits[:, -1:], form_logits], dim=1)
        for row, form in enumerate(forms):
            if fits[row]:
                totals[row] += word_then_boundary_probability(logits[row], form, boundary_ids)
    return totals


def word_then_boundary_probability(logits, form, boundary_ids):
    log_probs = torch.log_softmax(logits[:len(form) + 1].double(), dim=-1)
    log_probability = sum(log_probs[i, token].item() for i, token in enumerate(form))
    boundary = torch.logsumexp(log_probs[len(form), boundary_ids], dim=0)
    return math.exp(log_probability + boundary.item())
