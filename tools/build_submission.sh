#!/usr/bin/env bash
# Assemble the platform archive:  dist/submission/  (pyproject.toml, README.md, src/, scripts/, model/)  and
# dist/guardian_submission.zip (stored, no compression; files at the archive top level, < 40 GB).
#   MODEL_SRC=/path/to/Qwen3.6-35B-A3B-FP8 bash tools/build_submission.sh
# MODEL_SRC = a local copy of the FP8 checkpoint (optionally stripped with experiments/strip_weights.py).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$HERE/dist}
SUB=$OUT/submission
: "${MODEL_SRC:?set MODEL_SRC to the model directory}"
rm -rf "$SUB" && mkdir -p "$SUB/model"
cp -rL "$MODEL_SRC"/. "$SUB/model/"
rm -rf "$SUB/model/.cache"
cp -r "$HERE/pyproject.toml" "$HERE/README.md" "$HERE/src" "$HERE/scripts" "$SUB/"
find "$SUB/src" -name __pycache__ -prune -exec rm -rf {} +
rm -rf "$SUB"/src/*.egg-info
cp "$HERE/model/calibration.json" "$SUB/model/"
SIZE=$(du -sb "$SUB" | cut -f1)
echo "submission dir: $SIZE bytes"
[ "$SIZE" -lt 40000000000 ] || { echo "TOO BIG: $SIZE"; exit 1; }
if [ "${NO_ZIP:-0}" != "1" ]; then
  rm -f "$OUT/guardian_submission.zip"
  (cd "$SUB" && zip -0 -r -q "$OUT/guardian_submission.zip" .)
  ls -la "$OUT/guardian_submission.zip"
fi
