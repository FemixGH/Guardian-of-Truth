#!/usr/bin/env bash
# Pre-send check of a candidate directory (dist/<name>: pyproject.toml, src/, scripts/, model/) without a GPU:
# fresh venv + `pip install .` exactly as the platform does, compile, run predict.py --no-llm on valid.parquet with a
# .parquet output, check the output schema/ids, show the calibration, size of model/ (weights resolved via symlinks).
#   VALID=/path/to/valid.parquet bash tools/precheck.sh dist/submission
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CAND=$(cd "${1:-$HERE/dist/submission}" && pwd)
: "${VALID:?set VALID to a labelled parquet (e.g. the competition valid.parquet)}"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
echo "== candidate $CAND"
rsync -a --exclude model --exclude __pycache__ "$CAND/" "$W/sub/"
mkdir -p "$W/sub/model" && cp "$CAND/model/calibration.json" "$W/sub/model/"
[ -d "$CAND/model/lora_opus47" ] && cp -r "$CAND/model/lora_opus47" "$W/sub/model/"
python3 -m venv "$W/venv" >/dev/null
"$W/venv/bin/pip" install -q "$W/sub" >/dev/null 2>&1 && echo "pip install . OK" || { echo "pip install FAILED"; exit 1; }
"$W/venv/bin/python" -m py_compile "$W"/sub/src/guardian/*.py "$W/sub/scripts/predict.py" && echo "compile OK"
(cd "$W/sub" && "$W/venv/bin/python" scripts/predict.py --input "$VALID" --output "$W/out/pred.parquet" \
   --no-llm > "$W/run.log" 2>&1) && echo "predict.py --no-llm OK" || { tail -20 "$W/run.log"; exit 1; }
"$W/venv/bin/python" - "$VALID" "$W/out/pred.parquet" <<'E'
import sys
import pandas as pd
src, out = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
assert list(out.columns) == ["id", "label"], out.columns
assert len(out) == len(src) and list(out.id) == list(src.id), "ids/order differ"
assert set(out.label.unique()) <= {0, 1}, out.label.unique()
print(f"output OK: {len(out)} rows, dtypes {dict(out.dtypes.astype(str))}, positives (fallback only) {int(out.label.sum())}")
E
echo "== calibration"; python3 -c "import json,sys; c=json.load(open('$CAND/model/calibration.json')); print({k: c[k] for k in ('prompts','hard_flags','specialists','skip_flagged','layout') if k in c}, 'stop_at_verdict', c.get('stop_at_verdict'), 'hybrid', c.get('hybrid'))"
echo "== prompts referenced exist"
python3 -c "
import json; c=json.load(open('$CAND/model/calibration.json'))
import os
for p in c['prompts'] + list((c.get('specialists') or {}).values()):
    print(p, os.path.exists('$CAND/src/guardian/' + p))"
echo "== model dir (weights via symlinks)"; du -shL "$CAND/model"; find -L "$CAND/model" -name "*.safetensors" | wc -l
echo PRECHECK_OK
