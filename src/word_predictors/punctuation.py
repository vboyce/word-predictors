"""Punctuation shared by the n-gram and unigram predictors. The characters are
those kid-maze-passages strips for word length and frequency (hyphens kept)."""

PUNCTUATION = ".,;:!?\"'()‘’“”"
# Not word boundaries: they are usually inside a word ("cat's", "don't").
APOSTROPHES = "'’"


def strip_punctuation(word):
    return word.strip(PUNCTUATION)
