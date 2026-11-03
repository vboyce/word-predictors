#!/usr/bin/env bash
# Download a training corpus into corpora/<name>/, pinned to a revision and
# checked against its sha256. Refuses to overwrite an existing folder.
# Usage: corpora/download.sh tinystories
# Bandwidth is limited (--limit-rate) so the download doesn't swamp the network.
set -euo pipefail
cd "$(dirname "$0")"

download() {  # url, file name, sha256
    curl --fail --location --limit-rate 5M --output "$2" "$1"
    echo "$3  $2" | sha256sum --check
}

case "${1:?usage: download.sh <corpus name>}" in
tinystories)
    # The original TinyStories, which roneneldan/TinyStories-* models were trained on
    # (dataset card); not TinyStoriesV2-GPT4.
    revision=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
    base="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/$revision"
    mkdir tinystories  # fails if it already exists
    cd tinystories
    download "$base/TinyStories-train.txt" TinyStories-train.txt \
        c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f
    download "$base/TinyStories-valid.txt" TinyStories-valid.txt \
        94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4
    ;;
childes)
    # English CHILDES as prepared by Feng et al. (2024), the training data of
    # mcxfrank/childes-gpt2-ladder (its model card). Each ladder model saw a
    # shuffled prefix of it, not all of it.
    commit=1ba0423a20ec5609ea7f34b71d20c783736c3314
    mkdir childes  # fails if it already exists
    cd childes
    download "https://media.githubusercontent.com/media/styfeng/TinyDialogues/$commit/data/CHILDES_data.zip" \
        CHILDES_data.zip b05809e55dee7434baf989e827198474d871126d06893c778fa94c689730c59f
    unzip -q CHILDES_data.zip
    ;;
*)
    echo "unknown corpus: $1" >&2
    exit 1
    ;;
esac
