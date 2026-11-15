# Training and evaluation

This directory provides the rLLM/VERL GSPO pipeline for MobileGym GUI agents.
Browser rollouts use `bench_env` and its native state judge for rewards.
Run commands from the repository root unless noted otherwise.

## Runtime

The accelerator configuration targets ZW810E PPU with Python 3.12.3,
Torch `2.10.0+v0.1.0.ppu2.1.0.oe`, vLLM `0.19.0+v0.1.0.ppu2.1.0.oe`,
Transformers 5.6.2, tokenizers 0.22.2, Playwright 1.62.0, and vendored VERL 0.7.1.
Other accelerator stacks require compatibility validation.

Install Torch, vLLM, torchvision, and FlashAttention for your hardware first.
Install `uv`, nginx, and OpenSSL, then activate that Python environment and run:

```bash
cd mobilegym-rl
bash install.sh
cd ..
npm ci
npm run build
```

`install.sh` installs the remaining runtime and local packages without replacing
the accelerator wheels. Optional `RL_VENV=/path/to/venv` selects a virtual
environment. The shared input gate checks the Processor/VERL boundary before training.

## Train

```bash
export MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
export MOBILEGYM_DATA_ROOT=/path/to/output-data
# Set CUDA_VISIBLE_DEVICES to the devices available on your machine.
export TRAIN_CONFIG="$PWD/configs/training/feature0-selective.json"

# Inspect configuration without launching services or training.
MOBILEGYM_CONFIG_ONLY=1 RUN_NAME=lite_check \
bash mobilegym-rl/cookbooks/mobilegym/run_qwen3_vl_4b_env_effect_v3_full.sh

# Start training and its local simulator gateway.
RUN_NAME=lite_train \
bash mobilegym-rl/cookbooks/mobilegym/run_qwen3_vl_4b_env_effect_v3_full.sh
```

Choose `full.json`, `feature0.json`, or `feature0-selective.json` in
`configs/training/`. Use a new `RUN_NAME` for each independent run. When
`MOBILEGYM_DATA_ROOT` is unset, outputs go to the ignored root `artifacts/` folder.
Shell environment settings override JSON defaults; trailing Hydra arguments
override JSON overrides.

| Setting | Default |
| --- | --- |
| Train / evaluation templates | `env_effect_v3_train` (192) / `env_effect_v3_test` (155) |
| Devices | 16; override `CUDA_VISIBLE_DEVICES` for your hardware |
| Batch / rollouts / epochs | 12 tasks / 8 rollouts per task / 1 epoch |
| Parameter and optimizer storage / computation | FP32 / BF16 |
| Learning rate | 1e-6 |
| GSPO clip low / high; episode weighting | 0.2 / 0.28; `response_mean` |
| Training temperature / response tokens | 0.7 / 1024 |
| Task / validation / data seed | 41 / 42 / 42 |
| Simulator clock | `benchmark_time_ms=1789099200000` |
| Validation environment | Full (`val_completeness=null`) |

The gateway uses port 4180. Optional settings are `NGINX_BIN`, `API_WORKERS`,
`GATEWAY_PORT`, and `MOBILEGYM_STATIC_DIR`. If changing the gateway port, pass a
matching `++env_url` while preserving the fixed clock query parameter.
See [feature pruning](../docs/feature-pruning.md) and the
[split guide](../bench_env/splits/ENV_EFFECT_V3.md) for environment and sampling rules.

## Resume and export

For an unfinished run, keep its name and configuration and set `RESUME_MODE=auto`.
To branch from a complete checkpoint, use a new run name,
`RESUME_MODE=resume_path`, and `RESUME_FROM_PATH=/path/to/global_step_N`.
Resuming requires model, optimizer, extra state, and loader files with the
original device count. A model-only export cannot resume optimizer state.

Automatic export is enabled by default. To export explicitly:

```bash
cd mobilegym-rl
PYTHONPATH=verl:$PYTHONPATH python cookbooks/mobilegym/export_latest_checkpoint.py \
  --checkpoint-dir /path/to/checkpoint-run \
  --global-step 16 --output-dir /path/to/export-run
cd ..
```

The output `global_step_16/` contains BF16 Hugging Face model shards and
processor/tokenizer files.

## Evaluate in Full

The evaluation entry starts a local vLLM server and simulator gateway. Use an
isolated checkout if another run is using those services.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/export-run/global_step_16 RUN_NAME=lite_eval \
EVAL_SPLIT=env_effect_v3_test EVAL_COMPLETENESS_MODES=off \
EVAL_ENV_URL='https://127.0.0.1:4180/?benchmark_time_ms=1789099200000' \
bash mobilegym-rl/cookbooks/mobilegym/eval_mobilegym_4ppu_3runs.sh
```

Defaults are four inference devices and task seeds 42, 43, 44. Match
`VLLM_DATA_PARALLEL_SIZE` and `VLLM_TENSOR_PARALLEL_SIZE` to your device allocation.
Add `MOBILEGYM_CONFIG_ONLY=1` to inspect settings without launching services.
Use the same model, frontend build, runtime, sampling, and reward settings across
environments.

## Checks

With training dependencies installed:

```bash
PYTHONPATH="$PWD/mobilegym-rl/verl:$PWD/mobilegym-rl:$PWD" \
python -m pytest mobilegym-rl/tests -q
```

For simulator and feature-gate checks, see the
[root README](../README.md#verify-the-release) and
[feature guide](../docs/feature-pruning.md#checks).
