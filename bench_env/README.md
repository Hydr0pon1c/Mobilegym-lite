# MobileGym benchmark

`bench_env` runs GUI agents against the simulator and evaluates tasks using
structured state. Agent adapters, tasks, environments, and judges are separate
interfaces. The real-device backend supports Android and VLM judging.

## Setup

Start the simulator using the [root README](../README.md#start-the-simulator).
Install `bench_env/requirements.txt` and Playwright Chromium. Set
`MODEL_BASE_URL`, `MODEL_API_KEY`, and `MODEL_NAME` for your agent endpoint.
Simulator keys in [`.env.example`](../.env.example) are separate and optional.

## Run tasks

From the repository root:

```bash
python -m bench_env.run --list --split env_effect_v3_test

python -m bench_env.run \
  --task-id wechat.ReadMyWxid \
  --agent generic_v2 --model-name "$MODEL_NAME" \
  --model-base-url "$MODEL_BASE_URL" --model-api-key "$MODEL_API_KEY" \
  --env-url http://127.0.0.1:3000 --headless

python -m bench_env.run \
  --split env_effect_v3_test --sample-n 1 --sample-seed 42 \
  --parallel 8 --isolation pages \
  --agent generic_v2 --model-name "$MODEL_NAME" \
  --model-base-url "$MODEL_BASE_URL" --model-api-key "$MODEL_API_KEY" \
  --env-url http://127.0.0.1:3000/?benchmark_time_ms=1789099200000 \
  --headless
```

The Lite workflow uses the fixed `env_effect_v3_train` / `env_effect_v3_test`
splits and evaluates in Full; see the [split guide](splits/ENV_EFFECT_V3.md).
For upstream MobileGym-Bench, use `--split test` (256 templates). Other selectors
include `--suite`, `--split train+test`, and `--split /path/to/task_ids.txt`.

## Sampling

`--sample-n` controls distinct parameter instances per template;
`--sample-seed` makes their selection reproducible. Low-capacity templates can
produce fewer than the requested number. `--repeat-n` repeats the same sampled
tasks; `--pass-k 1,4` reports pass@k when using four repeats.

Lite evaluation uses seeds 42, 43, 44, each with `--sample-n 1`. These are task
sampling seeds. Keep training repetitions separate when reporting variability.
For pruned-environment diagnostics, add `--completeness feature0` or
`--completeness feature0_selective_v1`; omit it for standard Full evaluation.

## Parallel execution

For concurrent rollouts, build and start the nginx gateway:

```bash
npm run build
bash scripts/server/start_nginx_gateway.sh
# Use --env-url 'https://127.0.0.1:4180/?benchmark_time_ms=1789099200000'
# Stop the gateway:
bash scripts/server/start_nginx_gateway.sh stop
```

nginx and OpenSSL must be installed. Use `--isolation pages`, one browser per
process, and at most eight pages per browser. Size concurrency to the model
server's capacity. See [known issues](docs/KNOWN_ISSUES.md) for TLS, process,
and kernel-limit details.

## Outputs and extensions

Each local run writes `meta.json`, `results.jsonl`, `summary.json`, error records,
and per-step trajectories under `runs/`. Use `--resume`, `--rerun`, or `--prune`
as documented in the [CLI reference](docs/REFERENCE.md). The dev server's
`http://127.0.0.1:3000/run_explorer.html` can inspect local runs.
Keep these generated outputs outside a source distribution.

| Task | Guide |
| --- | --- |
| Choose or integrate an agent | [Agent guide](../docs/guides/bench-an-agent.md) |
| Author a task | [Authoring](docs/task/TASK_AUTHORING_GUIDE.md), [contracts](docs/task/TASK_CODE_SPEC.md), [tests](docs/task/TASK_TESTING_GUIDE.md) |
| Extend an agent, environment, or runner | [Framework](docs/FRAMEWORK.md) |
| Grounded answers and action fields | [Grounded mode](docs/task/GROUNDED_MODE.md) |
| CLI flags and programmatic configuration | [Reference](docs/REFERENCE.md) |
| Feature pruning | [MobileGym-Lite](../docs/feature-pruning.md) |
| Training, checkpoint export, evaluation | [Training guide](../mobilegym-rl/README.md) |
