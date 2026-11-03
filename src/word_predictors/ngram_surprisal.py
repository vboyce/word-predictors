"""N-gram surprisal from corpus counts with Stupid Backoff (Brants et al. 2007),
as in Michaelov & Levy (2026).

Words are matched with their outer punctuation stripped, case-sensitively.
An n-gram's context does not cross a sentence boundary in the stories, nor a
line boundary in a corpus, so the first word of a sentence gets only its
unigram, the second at most a bigram, and so on.

Stupid Backoff scores are not normalized probabilities: S(w | context) is
count(context w) / count(context) if count(context w) > 0, else alpha times
S(w | context minus its first word); S(w) = count(w) / total words. Surprisal
is -log2 S. A word never seen in the corpus has no surprisal (None); its zero
counts are in the output.
"""

import math
from collections import defaultdict

from word_predictors.punctuation import strip_punctuation


def story_sentences(word_rows):
    """[(topic, sentence, [row, ...]), ...] with rows in reading order."""
    sentences = defaultdict(list)
    for row in word_rows:
        sentences[(row["topic"], int(row["sentence"]))].append(row)
    return [(topic, sentence, sorted(rows, key=lambda r: int(r["word_position"])))
            for (topic, sentence), rows in sentences.items()]


def bare_words(rows):
    words = []
    for row in rows:
        bare = strip_punctuation(row["word"])
        if not bare:
            raise ValueError(f"{row['topic']} {row['sentence']}.{row['word_position']}: "
                             f"word {row['word']!r} is only punctuation")
        words.append(bare)
    return words


def needed_ngrams(word_rows, max_order):
    """Every n-gram (as a tuple of words) up to max_order that ends at a story
    word without crossing a sentence boundary, with all its suffixes and contexts."""
    ngrams = set()
    for _, _, rows in story_sentences(word_rows):
        words = bare_words(rows)
        for end in range(1, len(words) + 1):
            for start in range(max(0, end - max_order), end):
                ngrams.add(tuple(words[start:end]))
                if end - start > 1:
                    ngrams.add(tuple(words[start:end - 1]))
    return ngrams


def count_ngrams(lines, ngrams):
    """({ngram: count in the corpus} for the given ngrams, total words). Corpus
    words are whitespace-separated with outer punctuation stripped; tokens that
    are only punctuation are dropped."""
    counts = dict.fromkeys(ngrams, 0)
    orders = sorted({len(ngram) for ngram in ngrams})
    total_words = 0
    for line in lines:
        words = [w for w in map(strip_punctuation, line.split()) if w]
        total_words += len(words)
        for order in orders:
            for start in range(len(words) - order + 1):
                ngram = tuple(words[start:start + order])
                if ngram in counts:
                    counts[ngram] += 1
    return counts, total_words


def ngram_surprisal(word_rows, counts, total_words, max_order, alpha):
    """One row per story word and order 1..max_order, with the n-gram used,
    its counts, and its Stupid Backoff surprisal (bits)."""
    results = []
    for topic, sentence, rows in story_sentences(word_rows):
        words = bare_words(rows)
        for position, row in enumerate(rows):
            for order in range(1, max_order + 1):
                ngram = tuple(words[max(0, position + 1 - order):position + 1])
                score = stupid_backoff(ngram, counts, total_words, alpha)
                results.append({
                    "topic": topic,
                    "sentence": sentence,
                    "word_position": int(row["word_position"]),
                    "word": row["word"],
                    "order": order,
                    "ngram": " ".join(ngram),
                    "ngram_count": counts[ngram],
                    "context_count": counts[ngram[:-1]] if len(ngram) > 1 else None,
                    "unigram_count": counts[ngram[-1:]],
                    "ngram_surprisal": None if score is None else -math.log2(score),
                    "total_words": total_words,
                    "alpha": alpha,
                })
    return results


def stupid_backoff(ngram, counts, total_words, alpha):
    """Stupid Backoff score of the last word of ngram given the words before
    it, or None if the word itself was never seen."""
    if len(ngram) == 1:
        return counts[ngram] / total_words if counts[ngram] > 0 else None
    if counts[ngram] > 0:
        return counts[ngram] / counts[ngram[:-1]]
    shorter = stupid_backoff(ngram[1:], counts, total_words, alpha)
    return None if shorter is None else alpha * shorter
