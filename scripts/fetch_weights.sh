#!/usr/bin/env bash
# Fetch the utod-picker-12b-v1 release bundle (adapter + retriever + picker.json) and verify it.
# The base model (google/gemma-4-12B-it) is pulled from the HF hub on first load; accept its
# licence on HF and `hf auth login` first.
set -euo pipefail
SRC="${PICKER_SRC:-gs://dasein-jev-data/release/utod-picker-12b-v1/utod-picker-12b-v1}"
DST="${1:-picker}"
mkdir -p "$DST"
gsutil -m cp -r "$SRC/*" "$DST/"
cd "$DST" && { sha256sum -c sha256sums.txt 2>/dev/null || shasum -a 256 -c sha256sums.txt; }
