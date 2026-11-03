"""Readers for the local training corpora (see corpora/README.md), yielding
the units that n-gram counts must not cross: utterances for CHILDES, lines
(paragraphs) for TinyStories."""

import re

# Feng et al. (2024)'s CHILDES files separate utterances with a literal
# backslash-n pair, not newlines, and start each with a speaker label.
CHILDES_UTTERANCE_SEPARATOR = "\\n\\n"
CHILDES_SPEAKER_LABEL = re.compile(r"^\*\*[A-Z0-9_]+\*\*:\s*")
TINYSTORIES_STORY_SEPARATOR = "<|endoftext|>"


def childes_utterances(path):
    """Utterances in a CHILDES_*.txt file (with speaker labels), labels removed."""
    with open(path) as corpus:
        for line_number, line in enumerate(corpus, start=1):
            for utterance in line.split(CHILDES_UTTERANCE_SEPARATOR):
                utterance = utterance.strip()
                if not utterance:
                    continue
                label = CHILDES_SPEAKER_LABEL.match(utterance)
                if label is None:
                    raise ValueError(f"{path}:{line_number}: utterance without a speaker label: "
                                     f"{utterance[:60]!r}")
                yield utterance[label.end():]


def tinystories_lines(path):
    """Non-empty lines of a TinyStories file, without the story separators."""
    with open(path) as corpus:
        for line in corpus:
            line = line.strip()
            if line and line != TINYSTORIES_STORY_SEPARATOR:
                yield line
