"""Tiny randomly initialized GPT-2s with tiny byte-level BPE tokenizers, so
tests can run a real causal LM without a download.

save_tiny_lm: the tokenizer uses GPT-2's scheme (byte-level BPE, a space is marked as Ġ on
the following token, <|endoftext|> is BOS and EOS, no special tokens added by
encode). It is trained on TRAINING_TEXT only, so common words there are one
token and other words (e.g. "Tyrannosaurus") are split into several.

save_enumerable_lm: the same scheme with only ENUMERABLE_VOCAB, small enough to
enumerate every string of a few tokens and compute expectations exactly. " ab"
has a canonical tokenization (Ġa b) and a non-canonical one (Ġ a b).
"""
import math

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoConfig, GPTNeoForCausalLM,
                          GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast)

TRAINING_TEXT = ["The cat sat. The dog ran. A cat ran. A dog sat."] * 20
EXTRA_MERGES = 40  # merges beyond the 256 byte tokens and <|endoftext|>
MAX_POSITIONS = 8
ENUMERABLE_VOCAB = {"<|endoftext|>": 0, "a": 1, "b": 2, ".": 3, "Ġ": 4, "Ġa": 5, "Ġb": 6}
ENUMERABLE_MERGES = [("Ġ", "a"), ("Ġ", "b")]


def make_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=256 + 1 + EXTRA_MERGES,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(TRAINING_TEXT, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<|endoftext|>", eos_token="<|endoftext|>"
    )


def make_enumerable_tokenizer():
    tokenizer = Tokenizer(models.BPE(vocab=ENUMERABLE_VOCAB, merges=ENUMERABLE_MERGES))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<|endoftext|>", eos_token="<|endoftext|>"
    )


def make_whitespace_stripping_tokenizer():
    """A tokenizer that strips leading whitespace from its input, so " cat"
    encoded alone is "cat", but in running text ("The cat") it is " cat"."""
    vocab = {"<s>": 0, "<unk>": 1, "The": 2, " cat": 3, "cat": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Strip()
    tokenizer.pre_tokenizer = pre_tokenizers.Split(" ", behavior="merged_with_next")
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<s>", eos_token="<s>", unk_token="<unk>"
    )


def save_tiny_lm(directory, architecture="gpt2"):
    """Save the trained tokenizer and a seeded random model to directory; return
    its path. architecture: "gpt2" (absolute positions), "gpt_neox" (Pythia's:
    rotary positions), or "gpt_neo" (TinyStories': local attention layers)."""
    savers = {"gpt2": save_random_gpt2, "gpt_neox": save_random_gpt_neox,
              "gpt_neo": save_random_gpt_neo}
    return savers[architecture](make_tokenizer(), directory)


def save_enumerable_lm(directory):
    """Save the enumerable tokenizer and a seeded random GPT-2 to directory; return its path."""
    return save_random_gpt2(make_enumerable_tokenizer(), directory)


def save_random_gpt2(tokenizer, directory):
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2, n_positions=MAX_POSITIONS,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    GPT2LMHeadModel(config).save_pretrained(directory)
    return str(directory)


def save_random_gpt_neox(tokenizer, directory):
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=MAX_POSITIONS,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    GPTNeoXForCausalLM(config).save_pretrained(directory)
    return str(directory)


def save_random_gpt_neo(tokenizer, directory):
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    config = GPTNeoConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_layers=2, num_heads=2,
        attention_types=[[["global", "local"], 1]], window_size=4,
        max_position_embeddings=MAX_POSITIONS,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    GPTNeoForCausalLM(config).save_pretrained(directory)
    return str(directory)


def hand_surprisal(model, context_ids, token_id):
    """Surprisal in bits of one token after context_ids, from a single forward pass."""
    with torch.no_grad():
        logits = model(torch.tensor([context_ids])).logits[0, -1]
    return -torch.log_softmax(logits, dim=-1)[token_id].item() / math.log(2)
