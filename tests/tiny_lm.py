"""A tiny randomly initialized GPT-2 with a tiny byte-level BPE tokenizer, so
tests can run a real causal LM without a download.

The tokenizer uses GPT-2's scheme (byte-level BPE, a space is marked as Ġ on
the following token, <|endoftext|> is BOS and EOS, no special tokens added by
encode). It is trained on TRAINING_TEXT only, so common words there are one
token and other words (e.g. "Tyrannosaurus") are split into several.
"""
import math

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

TRAINING_TEXT = ["The cat sat. The dog ran. A cat ran. A dog sat."] * 20
EXTRA_MERGES = 40  # merges beyond the 256 byte tokens and <|endoftext|>
MAX_POSITIONS = 8


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


def save_tiny_lm(directory):
    """Save the tokenizer and a seeded random GPT-2 to directory; return its path."""
    tokenizer = make_tokenizer()
    tokenizer.save_pretrained(directory)
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2, n_positions=MAX_POSITIONS
    )
    GPT2LMHeadModel(config).save_pretrained(directory)
    return str(directory)


def hand_surprisal(model, context_ids, token_id):
    """Surprisal in bits of one token after context_ids, from a single forward pass."""
    with torch.no_grad():
        logits = model(torch.tensor([context_ids])).logits[0, -1]
    return -torch.log_softmax(logits, dim=-1)[token_id].item() / math.log(2)
