"""Registers the bundled vLLM-0.19.1 port of K2-Horizon in every Python process of the vLLM server (API server and
engine-core workers inherit PYTHONPATH), because the platform runs vLLM 0.19.1 and does not install packages."""
import os

if os.environ.get("GUARDIAN_K2_PORT"):
    try:
        from vllm import ModelRegistry

        ModelRegistry.register_model("K2HorizonForCausalLM", "k2_horizon_port:K2HorizonForCausalLM")
    except Exception as e:  # noqa: BLE001
        print("[guardian] K2 port registration failed:", repr(e), flush=True)
