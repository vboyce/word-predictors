import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import word_predictors.unigram_surprisal as unigram_module
from reference_unigram_surprisal import reference_expected_occurrences
from tiny_lm import (ENUMERABLE_VOCAB, make_tokenizer, make_whitespace_stripping_tokenizer,
                     save_enumerable_lm, save_tiny_lm)
from word_predictors.contextual_surprisal import load_causal_lm
from word_predictors.unigram_surprisal import (complete_word_count, continuation_log_probabilities,
                                               expected_occurrences, first_token_log_probabilities,
                                               priority_sampled_occurrences,
                                               sample_strings, unigram_surprisal, word_boundaries)

EOS = ENUMERABLE_VOCAB["<|endoftext|>"]
WORD_START_IDS = {i for token, i in ENUMERABLE_VOCAB.items() if token.startswith("Ġ")}
# A word ends before a space-initial token, punctuation, or EOS.
RIGHT_BOUNDARY_IDS = WORD_START_IDS | {ENUMERABLE_VOCAB["."], EOS}
TOKEN_TEXT = {i: token.replace("Ġ", " ") for token, i in ENUMERABLE_VOCAB.items()}
MAX_TOKENS = 4
# Canonical tokenizations at the start of a string and after a space.
WORD_TOKENS = {
    "a": (["a"], ["Ġa"]),
    "ab": (["a", "b"], ["Ġa", "b"]),
    "ba": (["b", "a"], ["Ġb", "a"]),
}


@pytest.fixture(scope="module")
def enumerable_lm(tmp_path_factory):
    path = save_enumerable_lm(tmp_path_factory.mktemp("enumerable_lm"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    return tokenizer, model


def all_strings():
    """Every string the sampler can produce: ending at EOS, or MAX_TOKENS tokens without EOS."""
    strings = []

    def extend(prefix):
        for token in ENUMERABLE_VOCAB.values():
            string = prefix + [token]
            if token == EOS or len(string) == MAX_TOKENS:
                strings.append(string)
            else:
                extend(string)

    extend([])
    return strings


def string_probability(model, string):
    ids = [EOS] + string  # BOS is the same token as EOS
    with torch.no_grad():
        log_probs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), dim=-1)
    return math.exp(sum(log_probs[i - 1, ids[i]].item() for i in range(1, len(ids))))


def ids(tokens):
    return [ENUMERABLE_VOCAB[t] for t in tokens]


def literal_occurrences(string, word):
    """Occurrences of the word's canonical tokens in string, starting a word
    and followed by a right-boundary token."""
    start_form, space_form = (ids(form) for form in WORD_TOKENS[word])
    count = 0
    for position in range(len(string)):
        forms = [start_form, space_form] if position == 0 else [space_form]
        for form in forms:
            end = position + len(form)
            if (string[position:end] == form and end < len(string)
                    and string[end] in RIGHT_BOUNDARY_IDS):
                count += 1
    return count


def literal_word_count(string):
    """Words in string: spans from a word start (string start or space-initial
    token) to the next right-boundary token, which must be in the string, and
    whose text, after its leading space, starts with neither punctuation nor
    nothing."""
    count = 0
    for start, token in enumerate(string):
        if token == EOS or not (start == 0 or token in WORD_START_IDS):
            continue
        ends = [i for i in range(start + 1, len(string)) if string[i] in RIGHT_BOUNDARY_IDS]
        text = "".join(TOKEN_TEXT[t] for t in string[start:ends[0]]).lstrip() if ends else ""
        if text and text[0] != ".":
            count += 1
    return count


@pytest.fixture(scope="module")
def exact(enumerable_lm):
    """Every string with its probability, and the exact canonical unigram surprisal of each word."""
    _, model = enumerable_lm
    strings = all_strings()
    probabilities = [string_probability(model, s) for s in strings]
    expected_words = sum(p * literal_word_count(s) for s, p in zip(strings, probabilities))
    surprisals = {
        word: -math.log2(sum(p * literal_occurrences(s, word)
                             for s, p in zip(strings, probabilities)) / expected_words)
        for word in WORD_TOKENS
    }
    return strings, probabilities, surprisals


def test_enumerated_strings_have_total_probability_one(exact):
    _, probabilities, _ = exact
    assert sum(probabilities) == pytest.approx(1, abs=1e-9)


def test_estimate_converges_to_exact_canonical_unigram_surprisal(enumerable_lm, exact):
    tokenizer, model = enumerable_lm
    _, _, exact_surprisals = exact

    rows = unigram_surprisal(list(WORD_TOKENS), tokenizer, model,
                             n_samples=2000, max_tokens=MAX_TOKENS, seed=0)

    assert [r["word"] for r in rows] == list(WORD_TOKENS)
    for row in rows:
        # Bootstrap standard error at 2000 samples is about 0.02 bits; the
        # exact values are more than 3 bits apart.
        assert row["unigram_surprisal"] == pytest.approx(exact_surprisals[row["word"]], abs=0.06)


def test_output_records_sampling_settings(enumerable_lm):
    tokenizer, model = enumerable_lm
    (row,) = unigram_surprisal(["a"], tokenizer, model, n_samples=10, max_tokens=3, seed=7)
    assert (row["n_samples"], row["max_tokens"], row["seed"]) == (10, 3, 7)


@pytest.mark.parametrize("word", list(WORD_TOKENS))
def test_marginalized_count_is_unbiased(enumerable_lm, exact, word):
    tokenizer, model = enumerable_lm
    strings, probabilities, _ = exact

    per_string = expected_occurrences(strings, [word], tokenizer, model, MAX_TOKENS)[word]

    marginalized = sum(p * c for p, c in zip(probabilities, per_string))
    literal = sum(p * literal_occurrences(s, word) for s, p in zip(strings, probabilities))
    assert marginalized == pytest.approx(literal, rel=1e-6)


@pytest.mark.parametrize("string, expected", [
    ([EOS], 0),
    (ids(["a", "Ġb"]) + [EOS], 2),
    (ids(["a", "Ġb", "b"]), 1),          # cut off at MAX_TOKENS: the last word is incomplete
    (ids(["a", "Ġb", "b", "."]), 2),     # "bb" is complete: punctuation ends it
    (ids(["a", ".", "Ġb"]) + [EOS], 2),  # "a" and "b"
    (ids(["Ġ", "a", "b", "."]), 1),      # " ab", non-canonical tokens
    (ids(["Ġ", "."]) + [EOS], 0),        # a space and punctuation are not words
    (ids(["Ġ", "Ġa"]) + [EOS], 1),
    (ids([".", "a", "Ġb"]) + [EOS], 1),  # ".a" starts with punctuation
])
def test_complete_word_count(enumerable_lm, string, expected):
    tokenizer, _ = enumerable_lm
    boundaries = word_boundaries(tokenizer)
    assert complete_word_count(string, tokenizer, boundaries) == expected == literal_word_count(string)


def test_samples_follow_seed(enumerable_lm):
    _, model = enumerable_lm
    first = sample_strings(model, EOS, EOS, n_samples=50, max_tokens=MAX_TOKENS, seed=1)
    assert first == sample_strings(model, EOS, EOS, n_samples=50, max_tokens=MAX_TOKENS, seed=1)
    assert first != sample_strings(model, EOS, EOS, n_samples=50, max_tokens=MAX_TOKENS, seed=2)


def test_samples_end_at_eos_or_max_tokens(enumerable_lm):
    _, model = enumerable_lm
    strings = sample_strings(model, EOS, EOS, n_samples=200, max_tokens=MAX_TOKENS, seed=0)
    for string in strings:
        assert EOS not in string[:-1]
        assert string[-1] == EOS or len(string) == MAX_TOKENS
        assert len(string) <= MAX_TOKENS


@pytest.mark.parametrize("max_tokens", [0, 8])
def test_max_tokens_outside_context_window_raises(enumerable_lm, max_tokens):
    tokenizer, model = enumerable_lm  # context window 8: BOS plus up to 7 tokens
    with pytest.raises(ValueError, match="max_tokens"):
        unigram_surprisal(["a"], tokenizer, model, n_samples=10, max_tokens=max_tokens, seed=0)


def test_n_samples_below_one_raises(enumerable_lm):
    tokenizer, model = enumerable_lm
    with pytest.raises(ValueError, match="n_samples"):
        unigram_surprisal(["a"], tokenizer, model, n_samples=0, max_tokens=3, seed=0)


def test_tokenizer_without_space_marked_tokens_raises(enumerable_lm):
    _, model = enumerable_lm
    with pytest.raises(ValueError, match="scheme"):
        unigram_surprisal(["cat"], make_whitespace_stripping_tokenizer(), model,
                          n_samples=10, max_tokens=3, seed=0)


@pytest.mark.parametrize("bad_word", ["", "a b"])
def test_empty_or_whitespace_word_raises(enumerable_lm, bad_word):
    tokenizer, model = enumerable_lm
    with pytest.raises(ValueError, match="word"):
        unigram_surprisal([bad_word], tokenizer, model, n_samples=10, max_tokens=3, seed=0)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["gpt2", "EleutherAI/pythia-70m"])
def test_real_model_gives_frequent_word_lower_surprisal(model_name):
    tokenizer, model = load_causal_lm(model_name)

    rows = unigram_surprisal(["the", "dinosaurs"], tokenizer, model,
                             n_samples=4, max_tokens=12, seed=0)

    the, dinosaurs = (row["unigram_surprisal"] for row in rows)
    assert 0 < the < dinosaurs < math.inf


def test_right_boundaries_are_spaces_newlines_punctuation_and_eos_not_apostrophes():
    tokenizer = make_tokenizer()
    vocab = tokenizer.get_vocab()

    right_boundary_ids = word_boundaries(tokenizer).right_boundary_ids

    for token in ["Ġcat", "Ġ", "Ċ", ".", ",", "!", "\"", "<|endoftext|>"]:
        assert vocab[token] in right_boundary_ids, token
    for token in ["cat", "s", "'", "-"]:
        assert vocab[token] not in right_boundary_ids, token


def test_word_with_outer_punctuation_raises(enumerable_lm):
    tokenizer, model = enumerable_lm
    with pytest.raises(ValueError, match="punctuation"):
        unigram_surprisal(["a."], tokenizer, model, n_samples=10, max_tokens=3, seed=0)


def test_word_containing_a_word_boundary_raises(enumerable_lm):
    tokenizer, model = enumerable_lm
    with pytest.raises(ValueError, match="boundary"):
        unigram_surprisal(["a.b"], tokenizer, model, n_samples=10, max_tokens=3, seed=0)


# --- Fast scoring: same values as the original implementation ---

@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    path = save_tiny_lm(tmp_path_factory.mktemp("tiny_lm"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    return tokenizer, model


TINY_LM_WORDS = ["The", "cat", "sat", "dog", "cats", "sated"]  # the last two are several tokens
# The original implementation runs every form at every prefix, fitting or not,
# so BOS + max_tokens - 2 + the longest form (3 tokens) must fit in the window (8).
TINY_LM_MAX_TOKENS = 6


def plain_log_probability(model, context_ids, continuation_ids):
    """log P(continuation_ids | context_ids), from one ordinary forward pass."""
    ids = context_ids + continuation_ids
    with torch.no_grad():
        log_probs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), dim=-1)
    return sum(log_probs[len(context_ids) + i - 1, token].item()
               for i, token in enumerate(continuation_ids))


def test_first_token_log_probabilities_match_plain_forward_passes(tiny_lm):
    tokenizer, model = tiny_lm
    strings = [tokenizer.encode(" The cat sat."), tokenizer.encode(" A dog")]
    first_tokens = [tokenizer.encode(" cat")[0], tokenizer.encode(" sated")[0]]

    per_string = first_token_log_probabilities(model, tokenizer.bos_token_id, strings, first_tokens)

    for string, log_probs in zip(strings, per_string):
        assert log_probs.shape == (len(string) + 1, len(first_tokens))
        for prefix_length in range(len(string) + 1):
            context = [tokenizer.bos_token_id] + string[:prefix_length]
            for column, token in enumerate(first_tokens):
                assert log_probs[prefix_length, column].item() == pytest.approx(
                    plain_log_probability(model, context, [token]), abs=1e-5)


def test_continuation_log_probabilities_match_plain_forward_passes(tiny_lm):
    tokenizer, model = tiny_lm
    boundary_ids = sorted(word_boundaries(tokenizer).right_boundary_ids)
    string = tokenizer.encode(" The cat sat")
    placements = [(0, tokenizer.encode(" dog")), (1, tokenizer.encode(" sated")),
                  (3, tokenizer.encode(" cat")), (2, tokenizer.encode(" sated"))]

    log_probs = continuation_log_probabilities(model, tokenizer.bos_token_id, string,
                                               placements, boundary_ids)

    for (prefix_length, form), log_prob in zip(placements, log_probs.tolist()):
        context = [tokenizer.bos_token_id] + string[:prefix_length] + form[:1]
        expected = plain_log_probability(model, context, form[1:])
        with torch.no_grad():
            after = model(torch.tensor([context + form[1:]])).logits[0, -1].double()
        expected += torch.logsumexp(torch.log_softmax(after, dim=-1)[boundary_ids], dim=0).item()
        assert log_prob == pytest.approx(expected, abs=1e-5)


def test_continuations_split_across_passes_give_the_same_values(tiny_lm, monkeypatch):
    tokenizer, model = tiny_lm
    boundary_ids = sorted(word_boundaries(tokenizer).right_boundary_ids)
    string = tokenizer.encode(" The cat sat")
    placements = [(j, tokenizer.encode(word)) for j in range(4) for word in [" dog", " sated"]]
    together = continuation_log_probabilities(model, tokenizer.bos_token_id, string,
                                              placements, boundary_ids)

    monkeypatch.setattr(unigram_module, "CONTINUATION_TOKENS_PER_PASS", 3)
    split = continuation_log_probabilities(model, tokenizer.bos_token_id, string,
                                           placements, boundary_ids)

    assert split.tolist() == pytest.approx(together.tolist(), abs=1e-6)


@pytest.mark.parametrize("strings_per_sweep", [16, 5])  # one sweep batch, or several
def test_expected_occurrences_matches_original_implementation(tiny_lm, monkeypatch, strings_per_sweep):
    tokenizer, model = tiny_lm
    monkeypatch.setattr(unigram_module, "STRINGS_PER_SWEEP", strings_per_sweep)
    strings = sample_strings(model, tokenizer.bos_token_id, tokenizer.eos_token_id,
                             n_samples=12, max_tokens=TINY_LM_MAX_TOKENS, seed=3)
    strings += [tokenizer.encode(" The cat sat.") + [tokenizer.eos_token_id],
                tokenizer.encode("The dog ran. A cat")]

    fast = expected_occurrences(strings, TINY_LM_WORDS, tokenizer, model, TINY_LM_MAX_TOKENS)
    reference = reference_expected_occurrences(strings, TINY_LM_WORDS, tokenizer, model,
                                               TINY_LM_MAX_TOKENS)

    for word in TINY_LM_WORDS:
        assert fast[word] == pytest.approx(reference[word], rel=1e-5, abs=1e-12), word


def test_model_with_attention_that_ignores_custom_masks_raises(tiny_lm, monkeypatch):
    tokenizer, model = tiny_lm
    monkeypatch.setattr(model.config, "_attn_implementation", "flash_attention_2")
    with pytest.raises(ValueError, match="attention"):
        unigram_surprisal(["cat"], tokenizer, model, n_samples=2, max_tokens=3, seed=0)


# --- Priority-sampled continuations (Hopton et al. App. I, Alg. 3) ---

@pytest.fixture(scope="module")
def tiny_lm_strings(tiny_lm):
    tokenizer, model = tiny_lm
    return sample_strings(model, tokenizer.bos_token_id, tokenizer.eos_token_id,
                          n_samples=12, max_tokens=TINY_LM_MAX_TOKENS, seed=3)


@pytest.fixture(scope="module")
def tiny_lm_exact_totals(tiny_lm, tiny_lm_strings):
    tokenizer, model = tiny_lm
    per_string = expected_occurrences(tiny_lm_strings, TINY_LM_WORDS, tokenizer, model,
                                      TINY_LM_MAX_TOKENS)
    return {word: sum(totals) for word, totals in per_string.items()}


def test_priority_estimate_is_exact_when_budget_covers_every_prefix(
        tiny_lm, tiny_lm_strings, tiny_lm_exact_totals):
    tokenizer, model = tiny_lm

    estimate = priority_sampled_occurrences(tiny_lm_strings, TINY_LM_WORDS, tokenizer, model,
                                            TINY_LM_MAX_TOKENS, continuations_per_word=1000, seed=0)

    for word in TINY_LM_WORDS:
        assert estimate[word] == pytest.approx(tiny_lm_exact_totals[word], rel=1e-9), word


def test_priority_estimate_is_unbiased(tiny_lm, tiny_lm_strings, tiny_lm_exact_totals):
    tokenizer, model = tiny_lm
    n_seeds = 300
    # 2 of the ~60 prefixes per word: far from exact, so this checks the weights.
    estimates = [priority_sampled_occurrences(tiny_lm_strings, TINY_LM_WORDS, tokenizer, model,
                                              TINY_LM_MAX_TOKENS, continuations_per_word=2, seed=seed)
                 for seed in range(n_seeds)]

    for word in TINY_LM_WORDS:
        values = torch.tensor([estimate[word] for estimate in estimates], dtype=torch.float64)
        standard_error = values.std().item() / math.sqrt(n_seeds)
        assert standard_error > 0, word  # the budget really is smaller than the prefix count
        assert values.mean().item() == pytest.approx(tiny_lm_exact_totals[word],
                                                     abs=4 * standard_error), word


def test_priority_estimate_follows_seed(tiny_lm, tiny_lm_strings):
    tokenizer, model = tiny_lm

    def estimate(seed):
        return priority_sampled_occurrences(tiny_lm_strings, TINY_LM_WORDS, tokenizer, model,
                                            TINY_LM_MAX_TOKENS, continuations_per_word=2, seed=seed)

    assert estimate(1) == estimate(1)
    assert estimate(1) != estimate(2)


def test_priority_sampled_estimate_converges_to_exact_canonical_unigram_surprisal(enumerable_lm, exact):
    tokenizer, model = enumerable_lm
    _, _, exact_surprisals = exact

    rows = unigram_surprisal(list(WORD_TOKENS), tokenizer, model, n_samples=2000,
                             max_tokens=MAX_TOKENS, seed=0, continuations_per_word=200)

    for row in rows:
        assert row["unigram_surprisal"] == pytest.approx(exact_surprisals[row["word"]], abs=0.06)


def test_output_records_continuations_per_word(enumerable_lm):
    tokenizer, model = enumerable_lm
    (exact_row,) = unigram_surprisal(["a"], tokenizer, model, n_samples=10, max_tokens=3, seed=7)
    (sampled_row,) = unigram_surprisal(["a"], tokenizer, model, n_samples=10, max_tokens=3, seed=7,
                                       continuations_per_word=5)
    assert exact_row["continuations_per_word"] is None
    assert sampled_row["continuations_per_word"] == 5


def test_continuations_per_word_below_one_raises(enumerable_lm):
    tokenizer, model = enumerable_lm
    with pytest.raises(ValueError, match="continuations_per_word"):
        unigram_surprisal(["a"], tokenizer, model, n_samples=10, max_tokens=3, seed=0,
                          continuations_per_word=0)
