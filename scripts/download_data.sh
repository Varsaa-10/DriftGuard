#!/usr/bin/env bash
# Downloads the Cloud Log Sampling dataset (ArielCyber) and extracts the three JSON files into ./data
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/ArielCyber/cloud_log_sampling "$TMP/cls"
mkdir -p data
for z in TrainOn/base_a.zip TrainOn/train_labeled_a.zip TestOn/labeled_cloudtrail_a.zip; do
  unzip -o -q "$TMP/cls/$z" -d data
done
rm -rf "$TMP"
ls -la data
