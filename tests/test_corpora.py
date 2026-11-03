import pytest

from word_predictors.corpora import childes_utterances, tinystories_lines
from word_predictors.ngram_surprisal import count_ngrams

# The file format of Feng et al. (2024)'s CHILDES_train_ordered.txt: one
# conversation per line, utterances separated by a literal "\n\n" (backslash, n).
CHILDES_TEXT = (
    "**MOT**: just like your book \\n\\n **CHI**: book \\n\\n **MOT**: there's Pooh_Bear \\n\\n\n"
    "**FAT**: look \\n\\n\n"
)
TINYSTORIES_TEXT = "One day, Lily ran.\nShe was happy.\n<|endoftext|>\nTom sat.\n<|endoftext|>\n"


def write(tmp_path, text):
    path = tmp_path / "corpus.txt"
    path.write_text(text)
    return path


def test_childes_utterances_without_speaker_labels(tmp_path):
    utterances = list(childes_utterances(write(tmp_path, CHILDES_TEXT)))
    assert utterances == ["just like your book", "book", "there's Pooh_Bear", "look"]


def test_childes_ngrams_do_not_cross_utterances(tmp_path):
    counts, total_words = count_ngrams(childes_utterances(write(tmp_path, CHILDES_TEXT)),
                                       {("book", "book"), ("book",)})
    assert counts == {("book", "book"): 0, ("book",): 2}
    assert total_words == 8


def test_childes_utterance_without_speaker_label_raises(tmp_path):
    path = write(tmp_path, "**MOT**: look \\n\\n no label here \\n\\n\n")
    with pytest.raises(ValueError, match="speaker label"):
        list(childes_utterances(path))


def test_tinystories_lines_without_story_separators(tmp_path):
    lines = list(tinystories_lines(write(tmp_path, TINYSTORIES_TEXT)))
    assert lines == ["One day, Lily ran.", "She was happy.", "Tom sat."]
