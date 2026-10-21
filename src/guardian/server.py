"""Start / wait / stop a local vLLM OpenAI-compatible server (A100 inference)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request


def start_vllm(model_path, served_name, port, max_model_len=131072, gpu_util=0.92, extra_args=None, log_path=None,
               reasoning_parser=True, hybrid_prefix_cache=False, max_num_seqs=128, mtp_tokens=0):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", served_name,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_util),
        "--language-model-only",
        "--enable-prefix-caching",
        "--max-num-seqs", str(max_num_seqs),
    ]
    if reasoning_parser:
        cmd += ["--reasoning-parser", "qwen3"]
    if mtp_tokens:
        # multi-token-prediction head shipped in the Qwen3.5/3.6 checkpoints -> lossless speculative decoding
        cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": %d}' % int(mtp_tokens)]
    if hybrid_prefix_cache:
        # prefix caching for hybrid (Gated-DeltaNet) models is only effective in "align" mode (vLLM >= 0.17)
        cmd += ["--mamba-cache-mode", "align", "--max-num-batched-tokens", "16384"]
    if extra_args:
        cmd += list(extra_args)
    env = dict(os.environ)
    # tools installed next to the interpreter (ninja for JIT kernels, etc.) must be on PATH even if the venv
    # is not activated
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    out = open(log_path, "w") if log_path else None
    # own process group, so that stop_vllm() also kills the engine-core children (frees GPU memory on restart)
    return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if out else None, start_new_session=True,
                            env=env)


def wait_ready(port, timeout=900, proc=None):
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM not ready on :{port} within {timeout}s")


def stop_vllm(proc):
    if proc is None:
        return
    pgid = proc.pid  # start_new_session=True -> group id == leader pid, valid even if the leader already died
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=20)
            break
        except Exception:  # noqa: BLE001
            continue
    time.sleep(2)
