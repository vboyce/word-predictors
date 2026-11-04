import copy
import csv
import math
from pathlib import Path

import pytest
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPTNeoXConfig,
                          PretrainedConfig)

from tiny_lm import (MAX_POSITIONS, hand_surprisal, make_whitespace_stripping_tokenizer,
                     save_tiny_lm)
from word_predictors.contextual_surprisal import (contextual_surprisal, model_context_window,
                                                      load_causal_lm)

KID_MAZE_GPT2 = Path(__file__).parent / "data" / "kid_maze_gpt2_surprisal.csv"
TOLERANCE_BITS = 1e-4

STORY_ROWS = [
    {"topic": "pets", "sentence": "1", "word_position": "1", "word": "The"},
    {"topic": "pets", "sentence": "1", "word_position": "2", "word": "cat"},
    {"topic": "pets", "sentence": "1", "word_position": "3", "word": "sat."},
    {"topic": "pets", "sentence": "2", "word_position": "1", "word": "A"},
    {"topic": "pets", "sentence": "2", "word_position": "2", "word": "Tyrannosaurus"},
    {"topic": "pets", "sentence": "2", "word_position": "3", "word": "ran."},
    {"topic": "dogs", "sentence": "1", "word_position": "1", "word": "A"},
    {"topic": "dogs", "sentence": "1", "word_position": "2", "word": "dog"},
    {"topic": "dogs", "sentence": "1", "word_position": "3", "word": "ran."},
]


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    path = save_tiny_lm(tmp_path_factory.mktemp("tiny_lm"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    return tokenizer, model


def brute_force_story(words, tokenizer, model, window_stride, window=MAX_POSITIONS):
    """Expected (surprisal, n_tokens) per word, one forward pass per token.
    Windows of `window` tokens start every window_stride tokens; token i is
    scored in the first window that contains it."""
    word_tokens = [tokenizer.encode(w if i == 0 else " " + w) for i, w in enumerate(words)]
    ids = [tokenizer.bos_token_id] + [t for tokens in word_tokens for t in tokens]
    token_surprisals = [None]
    for i in range(1, len(ids)):
        window_start = max(0, math.ceil((i + 1 - window) / window_stride)) * window_stride
        token_surprisals.append(hand_surprisal(model, ids[window_start:i], ids[i]))
    expected, position = [], 1
    for tokens in word_tokens:
        expected.append((sum(token_surprisals[position:position + len(tokens)]), len(tokens)))
        position += len(tokens)
    return expected


def test_matches_brute_force_with_story_longer_than_context(tiny_lm):
    tokenizer, model = tiny_lm
    window_stride = 3

    rows = contextual_surprisal(STORY_ROWS, tokenizer, model, window_stride)

    assert [(r["topic"], r["sentence"], r["word_position"], r["word"]) for r in rows] == [
        (r["topic"], int(r["sentence"]), int(r["word_position"]), r["word"]) for r in STORY_ROWS
    ]
    for topic in ["pets", "dogs"]:
        words = [r["word"] for r in STORY_ROWS if r["topic"] == topic]
        expected = brute_force_story(words, tokenizer, model, window_stride)
        got = [(r["surprisal"], r["n_tokens"]) for r in rows if r["topic"] == topic]
        assert [n for _, n in got] == [n for _, n in expected]
        assert [s for s, _ in got] == pytest.approx([s for s, _ in expected], abs=TOLERANCE_BITS)


@pytest.mark.slow
def test_gpt2_reproduces_kid_maze_word_predictors():
    with open(KID_MAZE_GPT2, newline="") as f:
        kid_maze = list(csv.DictReader(f))
    tokenizer, model = load_causal_lm("gpt2")

    rows = contextual_surprisal(kid_maze, tokenizer, model, window_stride=512)

    assert len(rows) == len(kid_maze) == 606
    by_key = {(r["topic"], r["sentence"], r["word_position"]): r for r in rows}
    for expected in kid_maze:
        got = by_key[(expected["topic"], int(expected["sentence"]), int(expected["word_position"]))]
        assert got["word"] == expected["word"]
        assert got["n_tokens"] == int(expected["n_tokens"])
        assert got["surprisal"] == pytest.approx(float(expected["surprisal"]), abs=TOLERANCE_BITS)


def test_context_window_from_gpt2_style_config():
    assert model_context_window(GPT2Config(n_positions=1024)) == 1024


def test_context_window_from_neox_style_config():
    assert model_context_window(GPTNeoXConfig(max_position_embeddings=2048)) == 2048


def test_context_window_missing_from_config_raises():
    with pytest.raises(ValueError, match="context window"):
        model_context_window(PretrainedConfig())


def test_context_window_conflicting_in_config_raises():
    # A plain config: GPT2Config aliases max_position_embeddings to n_positions.
    config = PretrainedConfig(n_positions=1024, max_position_embeddings=2048)
    with pytest.raises(ValueError, match="context window"):
        model_context_window(config)


@pytest.mark.parametrize("window_stride", [0, MAX_POSITIONS])
def test_window_stride_outside_context_window_raises(tiny_lm, window_stride):
    tokenizer, model = tiny_lm
    with pytest.raises(ValueError, match="window_stride"):
        contextual_surprisal(STORY_ROWS, tokenizer, model, window_stride)


def test_tokenizer_without_bos_raises(tiny_lm):
    tokenizer, model = tiny_lm
    no_bos = copy.deepcopy(tokenizer)
    no_bos.bos_token = None
    with pytest.raises(ValueError, match="BOS"):
        contextual_surprisal(STORY_ROWS, no_bos, model, window_stride=3)


def test_word_by_word_tokenization_differing_from_whole_text_raises(tiny_lm):
    _, model = tiny_lm
    stripping = make_whitespace_stripping_tokenizer()
    rows = [
        {"topic": "t", "sentence": "1", "word_position": "1", "word": "The"},
        {"topic": "t", "sentence": "1", "word_position": "2", "word": "cat"},
    ]
    with pytest.raises(ValueError, match="tokeniz"):
        contextual_surprisal(rows, stripping, model, window_stride=3)


@pytest.mark.parametrize("bad_word", ["", "two words"])
def test_empty_or_whitespace_word_raises(tiny_lm, bad_word):
    tokenizer, model = tiny_lm
    rows = STORY_ROWS[:1] + [{**STORY_ROWS[1], "word": bad_word}]
    with pytest.raises(ValueError, match="word"):
        contextual_surprisal(rows, tokenizer, model, window_stride=3)


def test_duplicate_word_position_raises(tiny_lm):
    tokenizer, model = tiny_lm
    with pytest.raises(ValueError, match="duplicate"):
        contextual_surprisal(STORY_ROWS + STORY_ROWS[:1], tokenizer, model, window_stride=3)


def test_load_causal_lm_from_subfolder(tmp_path):
    save_tiny_lm(tmp_path / "seed0" / "rung1M")

    tokenizer, model = load_causal_lm(str(tmp_path), subfolder="seed0/rung1M")

    assert model.config.n_positions == MAX_POSITIONS
    assert not model.training
    assert tokenizer.bos_token == "<|endoftext|>"


def test_load_causal_lm_loads_half_precision_weights_in_float32(tmp_path):
    # Pythia is stored in fp16; transformers 5 would load it as fp16, whose CPU
    # results differ by up to 1.5 bits per word between processors (2026-11-04).
    save_tiny_lm(tmp_path)
    _, model = load_causal_lm(str(tmp_path))
    model.half().save_pretrained(tmp_path)

    _, reloaded = load_causal_lm(str(tmp_path))

    assert reloaded.dtype == torch.float32


def test_load_causal_lm_from_missing_subfolder_raises(tmp_path):
    save_tiny_lm(tmp_path / "seed0" / "rung1M")
    with pytest.raises(FileNotFoundError, match="no_such_rung"):
        load_causal_lm(str(tmp_path), subfolder="seed0/no_such_rung")


def test_context_window_shorter_than_config_matches_brute_force(tiny_lm):
    # e.g. TinyStories-33M: config says 2048 positions, trained with 512.
    tokenizer, model = tiny_lm

    rows = contextual_surprisal(STORY_ROWS, tokenizer, model, window_stride=2, context_window=4)

    words = [r["word"] for r in STORY_ROWS if r["topic"] == "pets"]
    expected = brute_force_story(words, tokenizer, model, window_stride=2, window=4)
    got = [r["surprisal"] for r in rows if r["topic"] == "pets"]
    assert got == pytest.approx([s for s, _ in expected], abs=TOLERANCE_BITS)


def test_context_window_longer_than_config_raises(tiny_lm):
    tokenizer, model = tiny_lm
    with pytest.raises(ValueError, match="context_window"):
        contextual_surprisal(STORY_ROWS, tokenizer, model, window_stride=2,
                             context_window=MAX_POSITIONS + 1)
