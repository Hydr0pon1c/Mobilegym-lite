#!/usr/bin/env bash
# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOBILEGYM_ROOT="${MOBILEGYM_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
if [[ -n "${RL_VENV:-}" ]]; then
  if [[ ! -f "$RL_VENV/bin/activate" ]]; then
    echo "Invalid RL_VENV: expected bin/activate" >&2
    exit 2
  fi
  source "$RL_VENV/bin/activate"
fi
python_bin="$(command -v python)"
if [[ -n "${EVAL_RECOVERY_PLAN:-}" ]]; then
  echo "EVAL_RECOVERY_PLAN is retired; use explicit EVAL_* settings" >&2
  exit 2
fi
if [[ -n "${EVAL_CONFIG:-}" ]]; then
  echo "Visual EVAL_CONFIG workflows are outside this release; use explicit EVAL_* settings." >&2
  exit 2
fi

: "${MODEL_PATH:?Set MODEL_PATH to the model served by vLLM}"
: "${RUN_NAME:?Set RUN_NAME to the common eval run-name prefix}"

MOBILEGYM_DATA_ROOT="${MOBILEGYM_DATA_ROOT:-$MOBILEGYM_ROOT/artifacts}"
EVAL_RUNS_DIR="${EVAL_RUNS_DIR:-$MOBILEGYM_DATA_ROOT/eval/runs}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-$MOBILEGYM_DATA_ROOT/eval/logs/$RUN_NAME}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_PATH%/}}"
MODEL_LABEL="${MODEL_LABEL##*/}"
MODEL_LABEL="$(
  printf '%s' "$MODEL_LABEL" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
)"
MODEL_LABEL="${MODEL_LABEL:-mobilegym-model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_LABEL}"
VLLM_STABILIZE_SECONDS="${VLLM_STABILIZE_SECONDS:-120}"
EVAL_PARALLEL="${EVAL_PARALLEL:-40}"
EVAL_PROCESSES="${EVAL_PROCESSES:-5}"
EVAL_BROWSERS="${EVAL_BROWSERS:-5}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MODE="${EVAL_MODE:-text}"
EVAL_ENV_URL="${EVAL_ENV_URL:-https://127.0.0.1:4180}"
EVAL_COMPLETENESS_MODES="${EVAL_COMPLETENESS_MODES:-off}"
EVAL_SAMPLE_N="${EVAL_SAMPLE_N:-1}"
EVAL_SAMPLE_SEEDS="${EVAL_SAMPLE_SEEDS:-42 43 44}"
EVAL_REPEAT_N="${EVAL_REPEAT_N:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-2048}"
EVAL_INFER_TIMEOUT="${EVAL_INFER_TIMEOUT:-300}"
EVAL_LOOP_DETECT="${EVAL_LOOP_DETECT:-8}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-40960}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_CHAT_TEMPLATE_KWARGS="${VLLM_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "$VLLM_CHAT_TEMPLATE_KWARGS" ]]; then
  VLLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi
API_WORKERS="${API_WORKERS:-16}"
MOBILEGYM_CONFIG_ONLY_VALUE="${MOBILEGYM_CONFIG_ONLY:-false}"

read -r -a SAMPLE_SEEDS <<< "$EVAL_SAMPLE_SEEDS"
if (( ${#SAMPLE_SEEDS[@]} == 0 )); then
  echo "EVAL_SAMPLE_SEEDS must contain at least one integer seed" >&2
  exit 2
fi
for seed in "${SAMPLE_SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid task sample seed: $seed" >&2
    exit 2
  fi
done
read -r -a COMPLETENESS_MODES <<< "$EVAL_COMPLETENESS_MODES"
if (( ${#COMPLETENESS_MODES[@]} == 0 )); then
  echo "EVAL_COMPLETENESS_MODES must contain at least one condition" >&2
  exit 2
fi
declare -A SEEN_COMPLETENESS_MODES=()
for mode in "${COMPLETENESS_MODES[@]}"; do
  case "$mode" in
    off|resources_full|resources_0|resources_1|resources_2|feature0|feature1|feature0_mix_v1|feature0_layout_v1|feature0_selective_v1|feature0_plus_v1|lite_v1) ;;
    *) echo "Invalid completeness condition: $mode" >&2; exit 2 ;;
  esac
  if [[ -n "${SEEN_COMPLETENESS_MODES[$mode]:-}" ]]; then
    echo "Duplicate completeness condition: $mode" >&2
    exit 2
  fi
  SEEN_COMPLETENESS_MODES[$mode]=1
done
for setting in EVAL_SAMPLE_N EVAL_REPEAT_N EVAL_PARALLEL EVAL_PROCESSES EVAL_BROWSERS \
  EVAL_MAX_TOKENS VLLM_TENSOR_PARALLEL_SIZE VLLM_DATA_PARALLEL_SIZE \
  VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_BATCHED_TOKENS VLLM_MAX_NUM_SEQS API_WORKERS; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer; got: ${!setting}" >&2
    exit 2
  fi
done
if [[ ! "$EVAL_LOOP_DETECT" =~ ^[0-9]+$ ]]; then
  echo "EVAL_LOOP_DETECT must be a nonnegative integer" >&2
  exit 2
fi
EVAL_STEP_ARGS=()
if [[ -n "$EVAL_MAX_STEPS" ]]; then
  if [[ ! "$EVAL_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_MAX_STEPS must be a positive integer" >&2
    exit 2
  fi
  EVAL_STEP_ARGS=(--max-steps "$EVAL_MAX_STEPS")
fi
if (( EVAL_PROCESSES != EVAL_BROWSERS || EVAL_PARALLEL % EVAL_BROWSERS != 0 \
  || EVAL_PARALLEL / EVAL_BROWSERS > 8 )); then
  echo "Use one browser per process and 1-8 pages per browser (parallel must divide evenly)" >&2
  exit 2
fi

case "$EVAL_MODE" in
  grounded|text) ;;
  *)
    echo "EVAL_MODE must be 'grounded' or 'text'; got: $EVAL_MODE" >&2
    exit 2
    ;;
esac

case "${MOBILEGYM_CONFIG_ONLY_VALUE,,}" in
  1|true|yes|on)
    echo "Eval model: path=$MODEL_PATH served_name=$SERVED_MODEL_NAME"
    echo "Eval config: split=$EVAL_SPLIT mode=$EVAL_MODE sample_n=$EVAL_SAMPLE_N seeds=${SAMPLE_SEEDS[*]} repeat_n=$EVAL_REPEAT_N temperature=$EVAL_TEMPERATURE top_p=$EVAL_TOP_P"
    echo "Eval environment: url=$EVAL_ENV_URL static_dir=${MOBILEGYM_STATIC_DIR:-$MOBILEGYM_ROOT/dist}"
    echo "Eval resource conditions (sequential, shared model server): ${COMPLETENESS_MODES[*]}"
    echo "Eval concurrency: per_seed=$EVAL_PARALLEL total=$((EVAL_PARALLEL * ${#SAMPLE_SEEDS[@]})) processes_per_seed=$EVAL_PROCESSES browsers_per_seed=$EVAL_BROWSERS"
    echo "vLLM config: tp=$VLLM_TENSOR_PARALLEL_SIZE dp=$VLLM_DATA_PARALLEL_SIZE max_model_len=$VLLM_MAX_MODEL_LEN max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS max_num_seqs=$VLLM_MAX_NUM_SEQS memory=$VLLM_GPU_MEMORY_UTILIZATION"
    echo "Generation config: max_tokens=$EVAL_MAX_TOKENS timeout=$EVAL_INFER_TIMEOUT chat_template_kwargs=$VLLM_CHAT_TEMPLATE_KWARGS"
    echo "Episode limits: max_steps=${EVAL_MAX_STEPS:-task_default} loop_detect=$EVAL_LOOP_DETECT"
    exit 0
    ;;
  0|false|no|off) ;;
  *)
    echo "Invalid MOBILEGYM_CONFIG_ONLY=$MOBILEGYM_CONFIG_ONLY_VALUE (expected true or false)" >&2
    exit 2
    ;;
esac

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export PPU_SDK=/usr/local/PPU_SDK
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

mkdir -p "$EVAL_RUNS_DIR" "$EVAL_LOG_DIR"

echo "Eval model: path=$MODEL_PATH served_name=$SERVED_MODEL_NAME"

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  cd "$MOBILEGYM_ROOT"
  bash scripts/server/start_nginx_gateway.sh stop >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$MOBILEGYM_ROOT/mobilegym-rl"
python -m cookbooks.mobilegym.serve_vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --api-key EMPTY \
  --trust-remote-code \
  --dtype bfloat16 \
  --default-chat-template-kwargs "$VLLM_CHAT_TEMPLATE_KWARGS" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$VLLM_DATA_PARALLEL_SIZE" \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":1}' \
  >"$EVAL_LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

VLLM_READY=0
for _ in $(seq 1 600); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    VLLM_READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready; see $EVAL_LOG_DIR/vllm.log" >&2
    tail -n 100 "$EVAL_LOG_DIR/vllm.log" >&2 || true
    exit 1
  fi
  sleep 2
done

if [[ "$VLLM_READY" -ne 1 ]]; then
  echo "Timed out after 20 minutes waiting for vLLM; see $EVAL_LOG_DIR/vllm.log" >&2
  exit 1
fi

echo "vLLM is healthy; waiting ${VLLM_STABILIZE_SECONDS}s before starting MobileGym"
sleep "$VLLM_STABILIZE_SECONDS"
if ! kill -0 "$VLLM_PID" 2>/dev/null \
  || ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "vLLM became unavailable during the stabilization wait" >&2
  tail -n 100 "$EVAL_LOG_DIR/vllm.log" >&2 || true
  exit 1
fi

# Start MobileGym after the expensive vLLM initialization, then verify both
# nginx/static assets and the API backend before any episode can be scheduled.
cd "$MOBILEGYM_ROOT"
NGINX_ERROR_LOG="$MOBILEGYM_ROOT/.nginx/logs/error.log"
API_GATEWAY_LOG="$MOBILEGYM_ROOT/.nginx/logs/api_gateway.log"
NGINX_ERROR_LOG_START=$(wc -c <"$NGINX_ERROR_LOG" 2>/dev/null || echo 0)
API_GATEWAY_LOG_START=$(wc -c <"$API_GATEWAY_LOG" 2>/dev/null || echo 0)
API_WORKERS="$API_WORKERS" PYTHON_BIN="$python_bin" \
  bash scripts/server/start_nginx_gateway.sh 4180

GATEWAY_READY=0
GATEWAY_ROOT_READY=0
GATEWAY_MAP_READY=0
GATEWAY_API_READY=0
for attempt in $(seq 1 60); do
  curl -kfsS https://127.0.0.1:4180/ >/dev/null 2>&1 \
    && GATEWAY_ROOT_READY=1 || GATEWAY_ROOT_READY=0
  curl -kfsS https://127.0.0.1:4180/cdn/map/vector/manifest.json >/dev/null 2>&1 \
    && GATEWAY_MAP_READY=1 || GATEWAY_MAP_READY=0
  if "$python_bin" -c \
    'import socket; s = socket.create_connection(("127.0.0.1", 4181), 2); s.close()' \
    >/dev/null 2>&1; then
    GATEWAY_API_READY=1
  else
    GATEWAY_API_READY=0
  fi

  if [[ "$GATEWAY_ROOT_READY$GATEWAY_MAP_READY$GATEWAY_API_READY" == "111" ]]; then
    GATEWAY_READY=1
    echo "MobileGym ready: root=1 map=1 api=1 (attempt=$attempt)"
    break
  fi
  if (( attempt == 1 || attempt % 5 == 0 )); then
    echo "Waiting for MobileGym: root=$GATEWAY_ROOT_READY map=$GATEWAY_MAP_READY api=$GATEWAY_API_READY (attempt=$attempt/60)"
  fi
  sleep 2
done

if [[ "$GATEWAY_READY" -ne 1 ]]; then
  echo "MobileGym did not become ready: root=$GATEWAY_ROOT_READY map=$GATEWAY_MAP_READY api=$GATEWAY_API_READY" >&2
  echo "New nginx log output:" >&2
  tail -c "+$((NGINX_ERROR_LOG_START + 1))" "$NGINX_ERROR_LOG" 2>/dev/null \
    | tail -n 100 >&2 || true
  echo "New API gateway log output:" >&2
  tail -c "+$((API_GATEWAY_LOG_START + 1))" "$API_GATEWAY_LOG" 2>/dev/null \
    | tail -n 100 >&2 || true
  exit 1
fi

# Gateway startup first cleans up prior runtime state.  Keep this second vLLM
# check close to eval launch so a lifecycle bug cannot turn the entire suite
# into zero-step APIConnectionError results.
if ! kill -0 "$VLLM_PID" 2>/dev/null \
  || ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "vLLM became unavailable while starting MobileGym; refusing to launch eval" >&2
  tail -n 100 "$EVAL_LOG_DIR/vllm.log" >&2 || true
  exit 1
fi

run_eval() {
  local trial="$1"
  local sample_seed="$2"
  local mode="$3"
  local label="$trial"
  local completeness_args=()
  if [[ "$mode" != off ]]; then
    completeness_args=(--completeness "$mode")
  fi
  if [[ "$mode" != off ]] || (( ${#COMPLETENESS_MODES[@]} > 1 )); then
    label="${mode}_${trial}"
  fi
  local trial_name="${RUN_NAME}_${label}"

  echo "[eval_${label}] Starting task sample seed ${sample_seed}, completeness=${mode}"
  cd "$MOBILEGYM_ROOT"
  python -m bench_env.run \
    --split "$EVAL_SPLIT" \
    --sample-n "$EVAL_SAMPLE_N" \
    --sample-seed "$sample_seed" \
    --repeat-n "$EVAL_REPEAT_N" \
    --agent generic_v2 \
    --model-base-url http://127.0.0.1:8000/v1 \
    --model-api-key EMPTY \
    --model-name "$SERVED_MODEL_NAME" \
    --temperature "$EVAL_TEMPERATURE" \
    --top-p "$EVAL_TOP_P" \
    --max-tokens "$EVAL_MAX_TOKENS" \
    --infer-timeout "$EVAL_INFER_TIMEOUT" \
    --device sim \
    --env-url "$EVAL_ENV_URL" \
    "${completeness_args[@]}" \
    --headless \
    --processes "$EVAL_PROCESSES" \
    --parallel "$EVAL_PARALLEL" \
    --browsers "$EVAL_BROWSERS" \
    --isolation pages \
    --loop-detect "$EVAL_LOOP_DETECT" \
    "${EVAL_STEP_ARGS[@]}" \
    --judge-mode state \
    --eval-mode "$EVAL_MODE" \
    --runs-dir "$EVAL_RUNS_DIR/$trial_name" \
    2>&1 | sed -u "s/^/[eval_${label}] /" | tee "$EVAL_LOG_DIR/${trial_name}.log"
}

echo "Eval task sample seeds: ${SAMPLE_SEEDS[*]}; repeats per instance: $EVAL_REPEAT_N"
echo "Episode concurrency per condition: $((EVAL_PARALLEL * ${#SAMPLE_SEEDS[@]}))"
echo "Eval resource conditions (sequential, shared model server): ${COMPLETENESS_MODES[*]}"

STATUS=0
for mode in "${COMPLETENESS_MODES[@]}"; do
  EVAL_PIDS=()
  for i in "${!SAMPLE_SEEDS[@]}"; do
    run_eval "$((i + 1))" "${SAMPLE_SEEDS[$i]}" "$mode" &
    EVAL_PIDS+=("$!")
  done
  for pid in "${EVAL_PIDS[@]}"; do
    if ! wait "$pid"; then
      STATUS=1
    fi
  done
  if (( STATUS != 0 )); then
    break
  fi
done

exit "$STATUS"
