import json
from pathlib import Path

import pytest

from word_predictors.infini_gram import (PILE_TRAIN_INDEX, InfiniGramError, infini_gram_counts,
                                         starts_new_word, total_word_count, whole_word_count,
                                         word_start_terms)

RECORDED = json.loads((Path(__file__).parent / "data" / "infini_gram_responses.json").read_text())


def recorded_post(payload):
    """Replays responses recorded from the live API (tests/data/infini_gram_responses.json)."""
    for entry in RECORDED:
        if entry["payload"] == payload:
            return entry["status"], entry["response"]
    raise AssertionError(f"no recorded response for {payload}")


def fake_post(responses):
    """A post function answering ntd queries from {query text: response}."""
    def post(payload):
        assert (payload["index"], payload["query_type"]) == (PILE_TRAIN_INDEX, "ntd")
        return 200, responses[payload["query"]]
    return post


def ntd_response(query_tokens, prompt_count, continuations, approx=False):
    return {
        "approx": approx,
        "prompt_cnt": prompt_count,
        "result_by_token_id": {str(i): {"token": token, "cont_cnt": count, "prob": count / prompt_count}
                               for i, (token, count) in enumerate(continuations.items())},
        "tokens": query_tokens,
    }


def test_whole_word_count_excludes_continuations_inside_a_word():
    # 10 occurrences: 3 continue the word ("catast..."), 1 is at a document end.
    response = ntd_response(["▁cat"], 10, {"▁the": 4, "ast": 3, ".": 2})
    assert whole_word_count(response) == 7


@pytest.mark.parametrize("token, expected", [
    ("▁cat", True), ("▁", True), ("<0x0A>", True), ("<0x09>", True), ("\r", True),
    (".", True), (",", True), ('".', True), ("...", True), ("</s>", True),
    ("ast", False), ("s", False), ("'s", False), ("';", False), ("’", False),
    ("-", False), ("<0xE2>", False),
])
def test_starts_new_word(token, expected):
    assert starts_new_word(token) is expected


def test_recorded_trigram_count():
    # 532 occurrences; the only continuation inside a word is "';" (once).
    counts, details = infini_gram_counts({("the", "cat", "sat")}, PILE_TRAIN_INDEX, post=recorded_post)
    assert counts == {("the", "cat", "sat"): 531}
    assert details[("the", "cat", "sat")] == {"token_sequence_count": 532, "approx": False}


def test_unseen_ngram_counts_zero():
    counts, _ = infini_gram_counts({("zzqx", "blorptastic")}, PILE_TRAIN_INDEX, post=recorded_post)
    assert counts == {("zzqx", "blorptastic"): 0}


def test_api_error_raises_with_the_query():
    with pytest.raises(InfiniGramError, match="Invalid index.*the cat|the cat.*Invalid index"):
        infini_gram_counts({("the", "cat")}, "v4_no_such_index", post=recorded_post)


def test_error_in_a_successful_response_raises():
    post = fake_post({"the cat": {"error": "engine busy"}})
    with pytest.raises(InfiniGramError, match="engine busy"):
        infini_gram_counts({("the", "cat")}, PILE_TRAIN_INDEX, post=post)


def test_query_not_starting_at_a_word_boundary_raises():
    post = fake_post({"cat": ntd_response(["cat"], 10, {"▁the": 10})})
    with pytest.raises(InfiniGramError, match="word boundary"):
        infini_gram_counts({("cat",)}, PILE_TRAIN_INDEX, post=post)


def test_sampled_distribution_is_flagged_approx():
    post = fake_post({"cat": ntd_response(["▁cat"], 5000, {"▁the": 4000, "ast": 1000}, approx=True)})
    counts, details = infini_gram_counts({("cat",)}, PILE_TRAIN_INDEX, post=post)
    assert counts == {("cat",): 4000}
    assert details[("cat",)] == {"token_sequence_count": 5000, "approx": True}


@pytest.mark.slow
def test_live_api_matches_recorded_response():
    counts, _ = infini_gram_counts({("the", "cat", "sat")}, PILE_TRAIN_INDEX)
    assert counts == {("the", "cat", "sat"): 531}


VOCAB = {"<unk>": 0, "▁the": 1, "▁(": 2, "▁": 3, "the": 4, "5": 5, "(": 6, "<0xE2>": 7,
         "▁▁": 8, "▁é": 9, "é": 10}


def test_word_start_terms_are_space_then_letter_or_digit():
    terms = word_start_terms(VOCAB)
    # "▁the", "▁é"; and a bare space before a token starting with a letter or digit
    assert sorted(terms) == sorted([[1], [9], [3, 4], [3, 5], [3, 10]])


def test_total_word_count_sums_the_counts_of_all_word_start_terms():
    term_counts = {(1,): 100, (9,): 7, (3, 4): 2, (3, 5): 30, (3, 10): 1}
    queries = []

    def post(payload):
        queries.append(payload)
        assert (payload["index"], payload["query_type"]) == (PILE_TRAIN_INDEX, "count")
        (clause,) = payload["query_ids"]  # one clause: an OR of terms
        return 200, {"count": sum(term_counts[tuple(term)] for term in clause), "approx": True}

    total = total_word_count(PILE_TRAIN_INDEX, VOCAB, post=post, terms_per_query=2, tokens_per_query=3)

    assert total == 140
    # The API limits a query's terms and tokens; a bare-space pair is two tokens.
    assert all(len(query["query_ids"][0]) <= 2 for query in queries)
    assert all(sum(len(term) for term in query["query_ids"][0]) <= 3 for query in queries)


def test_total_word_count_error_raises():
    def post(payload):
        return 400, {"error": "query too long"}

    with pytest.raises(InfiniGramError, match="query too long"):
        total_word_count(PILE_TRAIN_INDEX, VOCAB, post=post, terms_per_query=2, tokens_per_query=3)
