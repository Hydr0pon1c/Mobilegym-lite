# MobileGym-Lite

[中文](README_zh.md)

MobileGym-Lite is a browser-based mobile environment for training and evaluating
GUI agents. It combines the MobileGym simulator, state-based task judges, and
an rLLM/VERL GSPO training pipeline. Episode-local feature pruning removes
unneeded app functions while preserving task data, required functions, and judges.

| Environment | Training configuration | Feature mode |
| --- | --- | --- |
| Full | [full.json](configs/training/full.json) | `null` |
| feature0 | [feature0.json](configs/training/feature0.json) | `feature0` |
| MobileGym-Lite | [feature0-selective.json](configs/training/feature0-selective.json) | `feature0_selective_v1` |

All three configurations evaluate in Full. MobileGym-Lite uses
`feature0_selective_v1`; the internal `lite_v1` compatibility mode is different.
See [feature pruning](docs/feature-pruning.md) for the masks and runtime contract.

## Start the simulator

Requirements: Node.js 22 or newer and Python 3.12. From the repository root:

```bash
npm ci
python -m venv .venv
source .venv/bin/activate
python -m pip install -r bench_env/requirements.txt
python -m playwright install chromium
```

Download the upstream companion media and offline map cache, and extract
`mobilegym-data/` into the repository root:

```bash
curl -fL -o mobilegym-data.tar.gz \
  https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz
tar -xzf mobilegym-data.tar.gz
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:3000`. Optional simulator settings are in
[.env.example](.env.example). Agent model credentials are configured separately
through `--model-base-url` and `--model-api-key`.

## Reproduce the workflow

The released task split contains 192 training and 155 evaluation templates.
Use training seed 41 and evaluation seeds 42, 43, 44, with one instance per
template. Fixed memberships and sampling settings are documented in the
[split guide](bench_env/splits/ENV_EFFECT_V3.md).

The [training guide](mobilegym-rl/README.md) covers accelerator dependencies,
training, resume, export, and evaluation. To inspect the Lite configuration
without launching services or training:

```bash
MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct \
MOBILEGYM_CONFIG_ONLY=1 RUN_NAME=lite_check \
TRAIN_CONFIG="$PWD/configs/training/feature0-selective.json" \
bash mobilegym-rl/cookbooks/mobilegym/run_qwen3_vl_4b_env_effect_v3_full.sh
```

Set `MODEL_PATH` to your local model and `MOBILEGYM_DATA_ROOT` to your output
directory. Configure `CUDA_VISIBLE_DEVICES` for your hardware before training;
the launcher defaults to 16 devices. Keep the model, frontend build, runtime,
task sampling, and reward settings consistent when comparing environments.

## Verify the release

```bash
python scripts/check_release.py
npm test
npm run build
python -m pytest bench_env/tests/common/test_feature_completeness.py \
  bench_env/tests/common/test_completeness.py \
  bench_env/tests/common/test_env_effect_v3_split.py -q -m 'not live'
```

See the [documentation index](docs/README.md) for benchmark commands, platform
extension guides, browser checks, and [source ZIP packaging](docs/release.md).

## Attribution and licenses

Based on [MobileGym](https://github.com/Purewhiter/mobilegym),
[rLLM](https://github.com/rllm-org/rllm), and
[VERL](https://github.com/volcengine/verl). Code is Apache-2.0; app data and
companion snapshots have separate CC BY-NC 4.0 and third-party terms.
Keep [LICENSE](LICENSE), [LICENSE-DATA](LICENSE-DATA), [NOTICE](NOTICE),
[DISCLAIMER](DISCLAIMER.md), and [third-party notices](THIRD_PARTY_NOTICES.md)
when redistributing.
