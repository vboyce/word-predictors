import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_lm import ENUMERABLE_VOCAB, make_whitespace_stripping_tokenizer, save_enumerable_lm
from word_predictors.contextual_surprisal import load_causal_lm
from word_predictors.unigram_surprisal import (complete_word_count, expected_occurrences,
                                               sample_strings, unigram_surprisal)

EOS = ENUMERABLE_VOCAB["<|endoftext|>"]
WORD_START_IDS = {i for token, i in ENUMERABLE_VOCAB.items() if token.startswith("Ġ")}
MAX_TOKENS = 4
# Canonical tokenizations at the start of a string and after a space.
WORD_TOKENS = {
    "a": (["a"], ["Ġa"]),
    "ab": (["a", "b"], ["Ġa", "b"]),
    "ba.": (["b", "a", "."], ["Ġb", "a", "."]),
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
    and followed by a word-start token or EOS."""
    start_form, space_form = (ids(form) for form in WORD_TOKENS[word])
    count = 0
    for position in range(len(string)):
        forms = [start_form, space_form] if position == 0 else [space_form]
        for form in forms:
            end = position + len(form)
            if (string[position:end] == form and end < len(string)
                    and (string[end] == EOS or string[end] in WORD_START_IDS)):
                count += 1
    return count


def literal_word_count(string):
    """Words in string that are followed by a word-start token or EOS."""
    tokens = string[:-1] if string[-1] == EOS else string
    starts = [i for i, token in enumerate(tokens) if i == 0 or token in WORD_START_IDS]
    complete = len(starts) if string[-1] == EOS else len(starts) - 1
    return max(complete, 0)


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
    (ids(["a", "Ġb", "b", "."]), 1),     # cut off at MAX_TOKENS: the last word is incomplete
    (ids(["a", ".", "Ġb"]) + [EOS], 2),  # "a." and "b"
    (ids(["Ġ", "a", "b", "."]), 0),
])
def test_complete_word_count(enumerable_lm, string, expected):
    tokenizer, _ = enumerable_lm
    assert complete_word_count(string, tokenizer) == expected == literal_word_count(string)


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
