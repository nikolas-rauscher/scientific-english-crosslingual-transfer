#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
SUBPROJECT_ROOT="$PROJECT_ROOT/cross_lingual_transfer_multilingual"
DEFAULT_RAW_ROOT="/ds-slt/sci-LLM/scilons/unpaywall_texts_pq/unpaywall_texts_pq_3"
RAW_ROOT="${1:-$DEFAULT_RAW_ROOT}"

LANGS=(
  "deu_Latn"
  "jpn_Jpan"
  "spa_Latn"
  "rus_Cyrl"
  "pol_Latn"
  "por_Latn"
)

echo "Raw root: $RAW_ROOT"

if [[ ! -d "$RAW_ROOT" ]]; then
  echo "ERROR: raw root not found: $RAW_ROOT" >&2
  exit 1
fi

for lang in "${LANGS[@]}"; do
  src="$RAW_ROOT/$lang"
  dst_dir="$SUBPROJECT_ROOT/data/languages/$lang/raw"
  dst_link="$dst_dir/source"

  if [[ ! -d "$src" ]]; then
    echo "ERROR: missing language source dir: $src" >&2
    exit 1
  fi

  mkdir -p "$dst_dir"

  if [[ -L "$dst_link" ]]; then
    current_target="$(readlink "$dst_link")"
    if [[ "$current_target" == "$src" ]]; then
      echo "OK  $lang -> $dst_link (already linked)"
      continue
    fi
    rm "$dst_link"
  elif [[ -e "$dst_link" ]]; then
    echo "ERROR: target exists and is not a symlink: $dst_link" >&2
    exit 1
  fi

  ln -s "$src" "$dst_link"
  echo "NEW $lang -> $dst_link"
done

echo "Done."
