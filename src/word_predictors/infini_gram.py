"""Whole-word n-gram counts from the infini-gram API (https://infini-gram.io).

An infini-gram count is a count of a token sequence, so "cat" also counts the
start of "catastrophe". A whole-word count here is the token-sequence count
times the share of its continuations that start a new word, from one
next-token-distribution (ntd) query per n-gram. The ntd is exact when the
n-gram occurs at most 1000 times, and estimated from 1000 sampled occurrences
otherwise (flagged "approx"). Occurrences at a document end have no
continuation and count as whole words.

A continuation starts a new word if its token starts with whitespace (▁ or a
whitespace character or byte) or with punctuation other than an apostrophe, or
is </s>: the same right boundary as the unigram estimator. The left boundary
is a space: the query text is tokenized with a leading ▁, so occurrences
after a newline (no ▁) are not counted.

The Pile-train index is tokenized with Llama-2's tokenizer. No retries: any
API error stops the run with the query that failed.
"""

import json
import re
import urllib.error
import urllib.request

from word_predictors.punctuation import APOSTROPHES, PUNCTUATION

INFINI_GRAM_URL = "https://api.infini-gram.io/"
PILE_TRAIN_INDEX = "v4_piletrain_llama"
SPACE_MARKER = "▁"
END_OF_DOCUMENT = "</s>"
BYTE_TOKEN = re.compile(r"<0x([0-9A-F]{2})>")


class InfiniGramError(RuntimeError):
    pass


def post_json(payload):
    """POST payload to the infini-gram API; return (HTTP status, decoded JSON body)."""
    request = urllib.request.Request(
        INFINI_GRAM_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "word-predictors"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        # The API explains errors in a JSON body.
        return error.code, json.loads(error.read())


def infini_gram_counts(ngrams, index, post=post_json):
    """({ngram: whole-word count}, {ngram: {"token_sequence_count", "approx"}})
    for each ngram (a tuple of words)."""
    counts, details = {}, {}
    for ngram in sorted(ngrams):
        response = next_token_distribution(" ".join(ngram), index, post)
        counts[ngram] = whole_word_count(response)
        details[ngram] = {"token_sequence_count": response["prompt_cnt"], "approx": response["approx"]}
    return counts, details


def next_token_distribution(text, index, post):
    payload = {"index": index, "query_type": "ntd", "query": text}
    status, response = post(payload)
    if status != 200 or "error" in response:
        raise InfiniGramError(f"query {payload} failed (HTTP {status}): {response.get('error', response)}")
    if not response["tokens"][0].startswith(SPACE_MARKER):
        raise InfiniGramError(f"query {text!r} doesn't start at a word boundary: {response['tokens']}")
    return response


def whole_word_count(response):
    """Occurrences of the queried tokens not continued inside the same word."""
    inside_word = sum(continuation["cont_cnt"]
                      for continuation in response["result_by_token_id"].values()
                      if not starts_new_word(continuation["token"]))
    return response["prompt_cnt"] - inside_word


def starts_new_word(token):
    """Whether a Llama-2 token (as the API spells it) ends the word before it."""
    if token == END_OF_DOCUMENT:
        return True
    byte = BYTE_TOKEN.fullmatch(token)
    # Bytes from 0x80 up are pieces of non-ASCII characters: not treated as boundaries.
    first = chr(int(byte[1], 16)) if byte else token[0]
    if byte and ord(first) >= 0x80:
        return False
    return first == SPACE_MARKER or first.isspace() or (first in PUNCTUATION and first not in APOSTROPHES)


def word_start_terms(vocab):
    """Token sequences that start a word, as counted by whole_word_count's
    left boundary: a ▁-token whose next character is a letter or digit, or a
    bare ▁ followed by a token starting with one (how Llama-2 writes numbers)."""
    bare_space_id = vocab[SPACE_MARKER]
    terms = []
    for token, i in vocab.items():
        if token.startswith(SPACE_MARKER) and len(token) > 1 and token[1].isalnum():
            terms.append([i])
        elif not token.startswith(SPACE_MARKER) and token[0].isalnum():
            terms.append([bare_space_id, i])
    return terms


def total_word_count(index, vocab, post=post_json, terms_per_query=4, tokens_per_query=500):
    """Number of words in the index: the summed counts of word_start_terms.
    Each query counts an OR of at most terms_per_query terms and
    tokens_per_query tokens (the API's limits are 4 and 500); the terms can't
    overlap at a position, so the counts add up."""
    total = 0
    for batch in batches_within_limits(word_start_terms(vocab), terms_per_query, tokens_per_query):
        payload = {"index": index, "query_type": "count", "query_ids": [batch]}
        status, response = post(payload)
        if status != 200 or "error" in response:
            raise InfiniGramError(f"count query for terms {batch[0]}..{batch[-1]} failed "
                                  f"(HTTP {status}): {response.get('error', response)}")
        total += response["count"]
    return total


def batches_within_limits(terms, terms_per_batch, tokens_per_batch):
    batch, batch_tokens = [], 0
    for term in terms:
        if batch and (len(batch) == terms_per_batch or batch_tokens + len(term) > tokens_per_batch):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(term)
        batch_tokens += len(term)
    if batch:
        yield batch
