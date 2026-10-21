"""Format robustness: predict.py must emit id,label for every input row, even on garbage input."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_garbage_rows(tmp_path):
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "prompt": ["", None, "⟦SYSTEM⟧\n[AVAILABLE TOOLS]\n- foo — bar\n⟦USER⟧ hi", "x" * 100000],
        "response": ["", "⟦ASSISTANT · ход 1⟧\n\t→ TOOL_CALL foo: {bad json", "\t→ TOOL_CALL baz: {}", None],
    })
    inp, out = tmp_path / "in.csv", tmp_path / "out.csv"
    df.to_csv(inp, index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/predict.py"), "--input", str(inp), "--output", str(out),
                    "--no-llm"], check=True)
    res = pd.read_csv(out)
    assert list(res.columns) == ["id", "label"]
    assert res["id"].astype(str).tolist() == ["a", "b", "c", "d"]
    assert set(res["label"]) <= {0, 1}
    assert res.loc[2, "label"] == 1  # unavailable tool `baz`
