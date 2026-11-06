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

Computation: one forward pass per string gives every word's first-token
probability at every prefix. The rest of each word and its boundary token are
scored for many (prefix, word) pairs in one forward pass over the string, with
an attention mask that lets each word see only its own prefix. With
continuations_per_word=k, the rest of each word is scored at only k prefixes
per word, chosen by priority sampling (Hopton et al. App. I): an unbiased
estimate of the same sum, much cheaper for large n_samples.

Supports tokenizers that mark a preceding space on the following token, like
GPT-2's and Pythia's byte-level BPE ("Ġcat"); anything else raises.
"""

import math
from collections import namedtuple

import torch

from word_predictors.contextual_surprisal import model_context_window
from word_predictors.punctuation import APOSTROPHES, PUNCTUATION, strip_punctuation

SPACE_MARKER = "Ġ"
# Batch sizes bound memory, which is mostly float64 log-probabilities over the
# whole vocabulary: one row per string position, or per word token scored.
SAMPLES_PER_BATCH = 256
STRINGS_PER_SWEEP = 16
CONTINUATION_TOKENS_PER_PASS = 1024
# Attention implementations that apply a custom 4D attention mask (flash
# attention, for one, doesn't).
MASKABLE_ATTENTION = ("eager", "sdpa")
PRIORITY_SEED_OFFSET = 1_000_003

# Token ids that can start a word, and that end the word before them.
WordBoundaries = namedtuple("WordBoundaries", ["word_start_ids", "right_boundary_ids"])


def unigram_surprisal(words, tokenizer, model, n_samples, max_tokens, seed,
                      continuations_per_word=None):
    """Unigram surprisal (bits) of each word. Returns one dict per word, in
    order, with word, unigram_surprisal, and the sampling settings.

    continuations_per_word=None scores every word at every prefix. An integer k
    scores each word's later tokens and boundary at only k prefixes, chosen by
    priority sampling (see priority_sampled_occurrences)."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if continuations_per_word is not None and continuations_per_word < 1:
        raise ValueError(f"continuations_per_word must be None or at least 1, got {continuations_per_word}")
    window = model_context_window(model.config)
    if not 0 < max_tokens < window:
        raise ValueError(f"max_tokens must be between 1 and {window - 1} "
                         f"(BOS plus max_tokens must fit in the context window), got {max_tokens}")
    check_attention_takes_custom_masks(model)  # fail before sampling, not after
    boundaries = word_boundaries(tokenizer)
    for word in words:
        canonical_forms(word, tokenizer, boundaries)

    strings = sample_strings(model, tokenizer.bos_token_id, tokenizer.eos_token_id,
                             n_samples, max_tokens, seed)
    total_words = sum(complete_word_count(string, tokenizer, boundaries) for string in strings)
    if continuations_per_word is None:
        per_string = expected_occurrences(strings, words, tokenizer, model, max_tokens)
        occurrences = {word: sum(totals) for word, totals in per_string.items()}
    else:
        occurrences = priority_sampled_occurrences(strings, words, tokenizer, model, max_tokens,
                                                   continuations_per_word, seed)
    rows = []
    for word in words:
        rows.append({
            "word": word,
            "unigram_surprisal": -math.log2(occurrences[word] / total_words),
            "n_samples": n_samples,
            "max_tokens": max_tokens,
            "seed": seed,
            "continuations_per_word": continuations_per_word,
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
    check_attention_takes_custom_masks(model)
    boundaries = word_boundaries(tokenizer)
    forms = [canonical_forms(word, tokenizer, boundaries) for word in words]
    space_forms = [space_form for _, space_form in forms]
    boundary_ids = sorted(boundaries.right_boundary_ids)
    # The empty prefix is the same for every string.
    at_empty_prefix = empty_prefix_occurrences(forms, tokenizer, model, boundary_ids, max_tokens)
    prefixes = [prefix_tokens(string, tokenizer.eos_token_id) for string in strings]
    totals = {word: [] for word in words}
    for batch_start in range(0, len(prefixes), STRINGS_PER_SWEEP):
        batch = prefixes[batch_start:batch_start + STRINGS_PER_SWEEP]
        first_log_probs = first_token_log_probabilities(model, tokenizer.bos_token_id, batch,
                                                        [form[0] for form in space_forms])
        for prefix, string_first_log_probs in zip(batch, first_log_probs):
            prefix_lengths, word_indices = fitting_placements(len(prefix), space_forms, max_tokens)
            continuation = continuation_log_probabilities(
                model, tokenizer.bos_token_id, prefix,
                [(j, space_forms[w]) for j, w in zip(prefix_lengths, word_indices)], boundary_ids)
            first = string_first_log_probs[prefix_lengths, word_indices]
            per_word = at_empty_prefix.clone().index_add_(
                0, torch.tensor(word_indices, dtype=torch.long, device=model.device),
                torch.exp(first + continuation))
            for word, total in zip(words, per_word.tolist()):
                totals[word].append(total)
    return totals


def priority_sampled_occurrences(strings, words, tokenizer, model, max_tokens,
                                 continuations_per_word, seed):
    """{word: an unbiased estimate of the sum over strings of expected_occurrences},
    from Hopton et al. (2026)'s priority sampling (App. I, Alg. 3, one stratum).

    Every word's first-token probability s_b is computed at every prefix b, as
    in expected_occurrences. The rest of the word and its boundary, c_b, is
    computed only at continuations_per_word prefixes per word: those with the
    largest s_b / u_b, u_b ~ Uniform(0, 1]. Each kept prefix is weighted by
    max(s_b, tau), tau being the next-largest s_b / u_b (0 if every prefix is
    kept), which makes sum(max(s_b, tau) c_b) unbiased for sum(s_b c_b).
    Unlike Hopton et al.'s Approx. variant, every prefix counts and c_b includes
    the right boundary, so the estimand is the same as expected_occurrences'.
    The empty prefix, the same in every string, is computed exactly."""
    check_attention_takes_custom_masks(model)
    boundaries = word_boundaries(tokenizer)
    forms = [canonical_forms(word, tokenizer, boundaries) for word in words]
    space_forms = [space_form for _, space_form in forms]
    boundary_ids = sorted(boundaries.right_boundary_ids)
    at_empty_prefix = empty_prefix_occurrences(forms, tokenizer, model, boundary_ids, max_tokens)
    prefixes = [prefix_tokens(string, tokenizer.eos_token_id) for string in strings]
    kept = priority_sample_prefixes(prefixes, space_forms, tokenizer.bos_token_id, model, max_tokens,
                                    continuations_per_word, seed)
    totals = at_empty_prefix * len(strings)
    for string_index, (prefix_lengths, word_indices, weights) in kept.items():
        continuation = continuation_log_probabilities(
            model, tokenizer.bos_token_id, prefixes[string_index],
            [(j, space_forms[w]) for j, w in zip(prefix_lengths, word_indices)], boundary_ids)
        totals.index_add_(0, torch.tensor(word_indices, dtype=torch.long, device=model.device),
                          torch.tensor(weights, dtype=torch.float64, device=model.device)
                          * torch.exp(continuation))
    return dict(zip(words, totals.tolist()))


def priority_sample_prefixes(prefixes, forms, bos_id, model, max_tokens, sample_size, seed):
    """{string index: (prefix lengths, form indices, weights)}: for each form,
    the sample_size nonempty prefixes with the largest s / u, where s is
    P(form's first token | prefix), 0 where the form and its boundary don't fit,
    and u ~ Uniform(0, 1]. Each weight is max(s, tau), tau being the form's
    (sample_size + 1)-th largest s / u, or 0 if there are no more prefixes."""
    # Seeded apart from sampling, so the u's aren't the sampler's random numbers.
    generator = torch.Generator().manual_seed(seed + PRIORITY_SEED_OFFSET)
    form_lengths = torch.tensor([len(form) for form in forms], device=model.device)
    best_q, best_s, best_string, best_prefix = (
        torch.zeros(0, len(forms), dtype=dtype, device=model.device)
        for dtype in (torch.float64, torch.float64, torch.long, torch.long))
    for batch_start in range(0, len(prefixes), STRINGS_PER_SWEEP):
        batch = prefixes[batch_start:batch_start + STRINGS_PER_SWEEP]
        first_log_probs = first_token_log_probabilities(model, bos_id, batch, [form[0] for form in forms])
        for offset, (prefix, log_probs) in enumerate(zip(batch, first_log_probs)):
            # Row 0 is the empty prefix, computed exactly elsewhere.
            prefix_lengths = torch.arange(1, len(prefix) + 1, device=model.device)
            s = torch.exp(log_probs[1:])
            s[prefix_lengths[:, None] + form_lengths[None, :] + 1 > max_tokens] = 0
            u = 1 - torch.rand(s.shape, generator=generator, dtype=torch.float64)
            q = s / u.to(model.device)
            best_q, order = torch.cat([best_q, q]).topk(min(sample_size + 1, len(best_q) + len(q)), dim=0)
            best_s = torch.cat([best_s, s]).gather(0, order)
            best_string = torch.cat([best_string, torch.full_like(q, batch_start + offset, dtype=torch.long)]
                                    ).gather(0, order)
            best_prefix = torch.cat([best_prefix, prefix_lengths[:, None].expand_as(q)]).gather(0, order)
    if len(best_q) > sample_size:
        tau = best_q[sample_size]
        best_q, best_s, best_string, best_prefix = (
            tensor[:sample_size] for tensor in (best_q, best_s, best_string, best_prefix))
    else:
        tau = torch.zeros(len(forms), dtype=torch.float64, device=model.device)
    weights = torch.maximum(best_s, tau[None, :])
    kept = {}
    for rank, form_index in (best_s > 0).nonzero().tolist():  # s = 0 adds nothing
        prefix_lengths, form_indices, form_weights = kept.setdefault(
            best_string[rank, form_index].item(), ([], [], []))
        prefix_lengths.append(best_prefix[rank, form_index].item())
        form_indices.append(form_index)
        form_weights.append(weights[rank, form_index].item())
    return kept


def prefix_tokens(string, eos_id):
    """The tokens after which a word can start: the string without its final EOS."""
    return string[:-1] if string and string[-1] == eos_id else string


def fitting_placements(prefix_length, forms, max_tokens):
    """(prefix lengths, form indices) of every placement of a form after a
    nonempty prefix of the string, where the form and its boundary token fit in
    max_tokens."""
    prefix_lengths, form_indices = [], []
    for j in range(1, prefix_length + 1):
        for index, form in enumerate(forms):
            if j + len(form) + 1 <= max_tokens:
                prefix_lengths.append(j)
                form_indices.append(index)
    return prefix_lengths, form_indices


def empty_prefix_occurrences(forms, tokenizer, model, boundary_ids, max_tokens):
    """Float64 tensor: for each (start form, space form) pair, P(start form,
    then a boundary | BOS) + P(space form, then a boundary | BOS), counting only
    the forms that fit in max_tokens with their boundary token."""
    candidates = [form for pair in forms for form in pair]  # start, space, start, space, ...
    fitting = [index for index, form in enumerate(candidates) if len(form) + 1 <= max_tokens]
    (first,) = first_token_log_probabilities(model, tokenizer.bos_token_id, [[]],
                                             [candidates[i][0] for i in fitting])
    continuation = continuation_log_probabilities(model, tokenizer.bos_token_id, [],
                                                  [(0, candidates[i]) for i in fitting], boundary_ids)
    probabilities = torch.zeros(len(candidates), dtype=torch.float64, device=model.device)
    probabilities[fitting] = torch.exp(first[0] + continuation)
    return probabilities.view(len(forms), 2).sum(dim=1)


def first_token_log_probabilities(model, bos_id, strings, first_token_ids):
    """For each string, a float64 tensor of shape [len(string) + 1,
    len(first_token_ids)]: row j holds log P(token | BOS + string[:j])."""
    columns = torch.tensor(first_token_ids, dtype=torch.long, device=model.device)
    per_string = []
    for batch_start in range(0, len(strings), STRINGS_PER_SWEEP):
        batch = strings[batch_start:batch_start + STRINGS_PER_SWEEP]
        longest = max(len(string) for string in batch)
        # Right padding needs no attention mask: a causal LM's logits at a
        # position don't depend on later tokens.
        ids = torch.tensor([[bos_id] + string + [bos_id] * (longest - len(string)) for string in batch],
                           device=model.device)
        with torch.no_grad():
            logits = model(ids).logits
        for row, string in enumerate(batch):
            log_probs = torch.log_softmax(logits[row, :len(string) + 1].double(), dim=-1)
            per_string.append(log_probs[:, columns])
    return per_string


def continuation_log_probabilities(model, bos_id, string, placements, boundary_ids):
    """Float64 tensor: for each (prefix_length, form) in placements,
    log P(form[1:], then one of boundary_ids | BOS + string[:prefix_length], form[0])."""
    boundary_index = torch.tensor(boundary_ids, dtype=torch.long, device=model.device)
    chunks = []
    chunk, chunk_tokens = [], 0
    for placement in placements:
        form_tokens = len(placement[1])
        if chunk and chunk_tokens + form_tokens > CONTINUATION_TOKENS_PER_PASS:
            chunks.append(chunk)
            chunk, chunk_tokens = [], 0
        chunk.append(placement)
        chunk_tokens += form_tokens
    if chunk:
        chunks.append(chunk)
    if not chunks:
        return torch.zeros(0, dtype=torch.float64, device=model.device)
    return torch.cat([packed_continuation_pass(model, bos_id, string, chunk, boundary_index)
                      for chunk in chunks])


def packed_continuation_pass(model, bos_id, string, placements, boundary_index):
    """continuation_log_probabilities for placements, from one forward pass:
    the string, then every form, each form attending only to its own prefix
    of the string and to itself."""
    context = [bos_id] + string[:max(prefix_length for prefix_length, _ in placements)]
    ids = list(context)
    positions = list(range(len(context)))
    # For each token: attend to keys 0..last_context_key and block_start..itself.
    last_context_key = list(range(len(context)))
    block_start = [len(context) + sum(len(form) for _, form in placements)] * len(context)
    next_tokens, is_last = [], []
    for prefix_length, form in placements:
        start = len(ids)
        ids += form
        positions += range(prefix_length + 1, prefix_length + 1 + len(form))
        last_context_key += [prefix_length] * len(form)
        block_start += [start] * len(form)
        # Token i of the form predicts token i + 1; its last token predicts the boundary.
        next_tokens += form[1:] + [0]
        is_last += [False] * (len(form) - 1) + [True]
    device = model.device
    keys = torch.arange(len(ids), device=device)[None, :]
    queries = keys.T
    allowed = ((keys <= torch.tensor(last_context_key, device=device)[:, None])
               | ((keys >= torch.tensor(block_start, device=device)[:, None]) & (keys <= queries)))
    mask = torch.zeros(len(ids), len(ids), dtype=model.dtype, device=device)
    mask[~allowed] = torch.finfo(model.dtype).min
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device), attention_mask=mask[None, None],
                       position_ids=torch.tensor([positions], device=device)).logits[0, len(context):]
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    is_last = torch.tensor(is_last, device=device)
    token_log_probs = log_probs.gather(1, torch.tensor(next_tokens, device=device)[:, None])[:, 0]
    token_log_probs[is_last] = torch.logsumexp(log_probs[is_last][:, boundary_index], dim=1)
    placement_of_token = torch.cumsum(is_last, dim=0) - is_last.long()
    return torch.zeros(len(placements), dtype=torch.float64, device=device).index_add_(
        0, placement_of_token, token_log_probs)


def check_attention_takes_custom_masks(model):
    implementation = model.config._attn_implementation
    if implementation not in MASKABLE_ATTENTION:
        raise ValueError(
            f"model attention implementation {implementation!r} may ignore the custom attention "
            f"masks this estimator uses; load the model with attn_implementation in {MASKABLE_ATTENTION}")
