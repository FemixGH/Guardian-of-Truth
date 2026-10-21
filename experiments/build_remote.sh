#!/bin/bash
# Build a submission zip on the GPU server from a tested code dir + stripped weights.
#   bash build_remote.sh <code_dir> <out_dir> <name> <model_subdir=src_dir> [...]
# e.g. bash build_remote.sh /workspace/g/sub13 /workspace/g/dist13 guardian_submission13 gemma=/workspace/g/models/gemma_lm qwen=/workspace/g/models/awq_lm
set -e
CODE=$1; OUT=$2; NAME=$3; shift 3
rm -rf $OUT; mkdir -p $OUT/submission/model; S=$OUT/submission
cp -r $CODE/src $CODE/scripts $CODE/pyproject.toml $CODE/README.md $S/
find $S/src -name __pycache__ -prune -exec rm -rf {} +
rm -rf $S/src/*.egg-info $S/build
for m in "$@"; do sub=${m%%=*}; src=${m#*=}; cp -rL $src $S/model/$sub; rm -rf $S/model/$sub/.cache; done
cp $CODE/model/calibration.json $S/model/calibration.json
SIZE=$(du -sb $S | cut -f1); echo "submission dir bytes: $SIZE"
[ "$SIZE" -lt 40000000000 ] || { echo "TOO BIG"; exit 1; }
(cd $S && zip -0 -r -q $OUT/$NAME.zip .)
ls -la $OUT/$NAME.zip
T=$OUT/unz; rm -rf $T; mkdir -p $T; (cd $T && unzip -q $OUT/$NAME.zip -x 'model/*/*.safetensors')
(cd $T && /workspace/venv/bin/python scripts/predict.py --input /workspace/g/guardian/timing_431_labeled.parquet --output $T/p.csv --no-llm | tail -1)
md5sum $OUT/$NAME.zip > $OUT/$NAME.md5; cat $OUT/$NAME.md5
echo BUILD_OK
