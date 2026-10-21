# Experiments

These research scripts were written during the competition. They are kept as a record of what was tried (see
`docs/SOLUTION.md`, section 6). They are **not maintained**: many expect data files that are not in the repo
(`runs/`, competition parquet files) or a GPU server layout under `/workspace/g`.

| Script(s) | Purpose |
|---|---|
| `serve_local.sh` | llama.cpp server for a GGUF Qwen3.6 on a 16 GB GPU (MoE experts on CPU), for local development |
| `run_judge.py`, `analyze.py`, `pipeline_eval.py`, `calib_eval.py`, `cmp*.py`, `final_table.py` | Run the judge on a labelled parquet, cache the outputs, compute metrics, compare configs |
| `cal_*.json` | `calibration.json` of earlier submissions (sub11, cand12, ...) |
| `strip_weights.py` | Drop the vision tower and MTP tensors from a checkpoint |
| `build_remote.sh` | Build an archive on a GPU server |
| `apb_*.py`, `translate_apb.py`, `apb_manual_labels.json` | AgentProcessBench (τ²-bench, human step labels) → this task's format, Russian translation, manual re-labelling |
| `synth.py`, `synth_llm.py` | Synthetic error injection (rule-based and LLM-based) |
| `train_cls.py` | LoRA single-token classifier |
| `lora/` | GSPO/SFT training of the judge, rollouts, merging, and the `exp_*.sh` end-to-end experiments on rented GPUs |
| `k2port/` | Port of K2-Horizon-MoVA to vLLM 0.19.1, plus symmetric int4 and int8 quantisation |
| `pbase_*.py`, `recall_gate.py`, `verify_*.py`, `vote_eval.py`, `lever_eval.py`, `rules2_eval.py` | Rejected levers: baseline-probability gate, recall gates, BM25 verification, voting |
