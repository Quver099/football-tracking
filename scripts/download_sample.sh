#!/usr/bin/env bash
# Download one sample broadcast clip from the Kaggle "DFL - Bundesliga Data Shootout" competition.
# Prereqs:
#   1. pip install kaggle
#   2. Put your API token at ~/.kaggle/kaggle.json (Kaggle -> Settings -> Create New Token)
#   3. Open https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/rules
#      and click "I Understand and Accept" (otherwise you get a 403).
set -euo pipefail

CLIP="${1:-08fd33_4.mp4}"
mkdir -p data/raw
kaggle competitions download -c dfl-bundesliga-data-shootout -f "clips/${CLIP}" -p data/raw
# Kaggle sometimes wraps single files in a zip
if [ -f "data/raw/${CLIP}.zip" ]; then
  unzip -o "data/raw/${CLIP}.zip" -d data/raw && rm "data/raw/${CLIP}.zip"
fi
ls -lh data/raw
