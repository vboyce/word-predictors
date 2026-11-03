import math

import pytest

from word_predictors.ngram_surprisal import count_ngrams, needed_ngrams, ngram_surprisal

ALPHA = 0.4
CORPUS = ["the cat sat.", "the dog sat.", "a cat ran."]
TOTAL_WORDS = 9
STORY_ROWS = [
    {"topic": "t", "sentence": "1", "word_position": "1", "word": "the"},
    {"topic": "t", "sentence": "1", "word_position": "2", "word": "cat"},
    {"topic": "t", "sentence": "1", "word_position": "3", "word": "sat."},
    {"topic": "t", "sentence": "2", "word_position": "1", "word": "the"},
    {"topic": "t", "sentence": "2", "word_position": "2", "word": "dog"},
    {"topic": "t", "sentence": "2", "word_position": "3", "word": "ran!"},
    {"topic": "t", "sentence": "2", "word_position": "4", "word": "bird"},
]


def surprisal_of(rows, sentence, word_position, order):
    (row,) = [r for r in rows if (r["sentence"], r["word_position"], r["order"])
              == (sentence, word_position, order)]
    return row


def test_stupid_backoff_matches_hand_counts():
    ngrams = needed_ngrams(STORY_ROWS, max_order=3)
    counts, total_words = count_ngrams(CORPUS, ngrams)

    rows = ngram_surprisal(STORY_ROWS, counts, total_words, max_order=3, alpha=ALPHA)

    expected = {
        # "sat" after "the cat": the trigram, the bigram (1 of 2 "cat"), the unigram (2 of 9)
        (1, 3, 3): 0.0,
        (1, 3, 2): 1.0,
        (1, 3, 1): math.log2(9 / 2),
        # first word of a sentence: no context, whatever the order
        (2, 1, 3): math.log2(9 / 2),
        # "ran" after "the dog": no trigram, no bigram, so backs off twice
        (2, 3, 3): -math.log2(ALPHA * ALPHA * 1 / 9),
        (2, 3, 2): -math.log2(ALPHA * 1 / 9),
    }
    for (sentence, word_position, order), surprisal in expected.items():
        row = surprisal_of(rows, sentence, word_position, order)
        assert row["ngram_surprisal"] == pytest.approx(surprisal, abs=1e-12)


def test_context_does_not_cross_sentence_boundaries():
    # "the" starts sentence 2; "sat the" would be a bigram across the boundary.
    ngrams = needed_ngrams(STORY_ROWS, max_order=3)
    assert ("sat", "the") not in ngrams
    assert ("cat", "sat", "the") not in ngrams


def test_unseen_word_has_no_surprisal_and_zero_counts():
    counts, total_words = count_ngrams(CORPUS, needed_ngrams(STORY_ROWS, max_order=2))

    rows = ngram_surprisal(STORY_ROWS, counts, total_words, max_order=2, alpha=ALPHA)

    for order in [1, 2]:
        row = surprisal_of(rows, 2, 4, order)
        assert row["ngram_surprisal"] is None
        assert row["unigram_count"] == 0


def test_output_records_counts_and_settings():
    counts, total_words = count_ngrams(CORPUS, needed_ngrams(STORY_ROWS, max_order=2))

    rows = ngram_surprisal(STORY_ROWS, counts, total_words, max_order=2, alpha=ALPHA)

    row = surprisal_of(rows, 1, 3, 2)
    assert (row["topic"], row["word"], row["ngram"]) == ("t", "sat.", "cat sat")
    assert (row["ngram_count"], row["context_count"], row["unigram_count"]) == (1, 2, 2)
    assert (row["total_words"], row["alpha"]) == (TOTAL_WORDS, ALPHA)
    assert len(rows) == len(STORY_ROWS) * 2


def test_counts_strip_punctuation_and_stay_within_lines():
    ngrams = {("sat",), ("sat", "the"), ("cat", "sat"), ("ran",)}

    counts, total_words = count_ngrams(CORPUS, ngrams)

    assert counts == {("sat",): 2, ("sat", "the"): 0, ("cat", "sat"): 1, ("ran",): 1}
    assert total_words == TOTAL_WORDS


def test_counts_are_case_sensitive():
    counts, _ = count_ngrams(["The cat", "the cat"], {("The",), ("the",)})
    assert counts == {("The",): 1, ("the",): 1}


def test_corpus_tokens_that_are_only_punctuation_are_dropped():
    counts, total_words = count_ngrams(["the ... cat"], {("the", "cat")})
    assert (counts[("the", "cat")], total_words) == (1, 2)


def test_word_that_is_only_punctuation_raises():
    rows = STORY_ROWS[:1] + [{**STORY_ROWS[1], "word": "..."}]
    with pytest.raises(ValueError, match="punctuation"):
        needed_ngrams(rows, max_order=2)


def test_ngram_missing_from_counts_raises():
    counts, total_words = count_ngrams(CORPUS, needed_ngrams(STORY_ROWS, max_order=2))
    with pytest.raises(KeyError):
        ngram_surprisal(STORY_ROWS, counts, total_words, max_order=3, alpha=ALPHA)
