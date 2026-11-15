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
RL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOCAL_VERL_ROOT="${LOCAL_VERL_ROOT:-$RL_ROOT/verl}"
if [[ ! -f "$LOCAL_VERL_ROOT/verl/__init__.py" ]]; then
  echo "Invalid LOCAL_VERL_ROOT=$LOCAL_VERL_ROOT (expected verl/__init__.py)" >&2
  exit 2
fi
export PYTHONPATH="$LOCAL_VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Keep both native and visual launches on the same in-repository rLLM chain.
export PYTHONPATH="$RL_ROOT:$PYTHONPATH"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
: "${MODEL_PATH:?Set MODEL_PATH to a local model checkpoint directory}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_PATH%/}}"
MODEL_LABEL="${MODEL_LABEL##*/}"
MODEL_SLUG="$(
  printf '%s' "$MODEL_LABEL" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
)"
MODEL_SLUG="${MODEL_SLUG:-model}"
MODEL_CACHE_KEY="$(
  printf '%s' "$MODEL_SLUG" \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
)"
MODEL_CACHE_KEY="${MODEL_CACHE_KEY:-model}"
MODEL_TYPE="unknown"
if [[ -f "$MODEL_PATH/config.json" ]]; then
  MODEL_TYPE="$(python - "$MODEL_PATH/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    print(json.load(config_file).get("model_type", "unknown"))
PY
)"
fi
MOBILEGYM_DATA_ROOT="${MOBILEGYM_DATA_ROOT:-$RL_ROOT/../artifacts}"
TRAIN_ARTIFACT_ROOT="${TRAIN_ARTIFACT_ROOT:-$MOBILEGYM_DATA_ROOT/train}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
GROUP_SIZE="${GROUP_SIZE:-8}"
EPOCHS="${EPOCHS:-1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-env_effect_v3_train}"
SAMPLE_N="${SAMPLE_N:-1}"
TRAIN_SAMPLE_SEED="${TRAIN_SAMPLE_SEED:-41}"
VAL_SPLIT="${VAL_SPLIT:-env_effect_v3_test}"
EVAL_MODE="${EVAL_MODE:-text}"
VAL_SAMPLE_N="${VAL_SAMPLE_N:-1}"
VAL_SAMPLE_SEED="${VAL_SAMPLE_SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-true}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
PPO_CLIP_RATIO_LOW="${PPO_CLIP_RATIO_LOW:-0.2}"
PPO_CLIP_RATIO_HIGH="${PPO_CLIP_RATIO_HIGH:-0.28}"
# Must cover one complete prompt/response training sequence. The profile may
# raise it further so dynamic batching can pack short trajectories densely.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-34000}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.7}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0}"
SAVE_FREQ="${SAVE_FREQ:-8}"
MAX_ACTOR_CKPTS="${MAX_ACTOR_CKPTS:-null}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
AUTO_EXPORT_MODEL="${AUTO_EXPORT_MODEL:-true}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
EXPORT_MAX_SHARD_SIZE="${EXPORT_MAX_SHARD_SIZE:-5GB}"
TEST_FREQ="${TEST_FREQ:-0}"
ENV_POOL_SIZE="${ENV_POOL_SIZE:-$((TRAIN_BATCH_SIZE * GROUP_SIZE))}"
ENV_NUM_BROWSERS="${ENV_NUM_BROWSERS:-$ENV_POOL_SIZE}"
if ! [[ "$MAX_RESPONSE_LENGTH" =~ ^[0-9]+$ ]] || (( MAX_RESPONSE_LENGTH == 0 )); then
  echo "Invalid MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH (expected a positive integer)" >&2
  exit 2
fi
if ! [[ "$PPO_MAX_TOKEN_LEN_PER_GPU" =~ ^[0-9]+$ ]] \
  || (( PPO_MAX_TOKEN_LEN_PER_GPU < 32768 + MAX_RESPONSE_LENGTH )); then
  echo "PPO_MAX_TOKEN_LEN_PER_GPU must cover 32768 + MAX_RESPONSE_LENGTH; got $PPO_MAX_TOKEN_LEN_PER_GPU" >&2
  exit 2
fi
# VERL's regular AdamW optimizer does not keep FP32 master weights. Loading an
# actor in BF16 therefore also creates BF16 Adam moments. Small updates can
# disappear when written back to the model, so full-parameter training keeps
# FP32 parameter shards while FSDP continues to compute in BF16.
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"
case "${ACTOR_MODEL_DTYPE,,}" in
  32|fp32|float32) ;;
  *)
    echo "Full-parameter training requires ACTOR_MODEL_DTYPE=fp32; got $ACTOR_MODEL_DTYPE" >&2
    echo "VERL AdamW has no separate FP32 master weights, so lower-precision model storage loses small updates." >&2
    exit 2
    ;;
esac
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mobilegym-$MODEL_SLUG-gspo-full}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="$TRAIN_ARTIFACT_ROOT/logs/mobilegym/$EXPERIMENT_NAME"
LOG_DIR="$LOG_ROOT/$RUN_NAME"
CHECKPOINT_DIR="$TRAIN_ARTIFACT_ROOT/checkpoints/mobilegym/$EXPERIMENT_NAME/$RUN_NAME"
HYDRA_OUTPUT_DIR="$TRAIN_ARTIFACT_ROOT/outputs/mobilegym/$EXPERIMENT_NAME/$RUN_NAME"
MODEL_EXPORT_DIR="$TRAIN_ARTIFACT_ROOT/models/mobilegym/$EXPERIMENT_NAME/$RUN_NAME"
PERSIST_COMPILE_CACHE="${PERSIST_COMPILE_CACHE:-true}"
COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT:-$TRAIN_ARTIFACT_ROOT/cache/$MODEL_CACHE_KEY}"
COMPILE_CACHE_LOCAL_ROOT="${COMPILE_CACHE_LOCAL_ROOT:-/tmp/mobilegym_${MODEL_CACHE_KEY}_compile_cache}"
# Take a one-shot snapshot before the platform's two-hour utilization window expires,
# then snapshot again whenever the script exits normally or via SIGTERM/SIGINT.
COMPILE_CACHE_SNAPSHOT_DELAY="${COMPILE_CACHE_SNAPSHOT_DELAY:-4200}"
MOBILEGYM_CONFIG_ONLY_VALUE="${MOBILEGYM_CONFIG_ONLY:-false}"

IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS="${#visible_devices[@]}"

# An explicit source supports checkpoint branches without copying shards or
# changing the source run's mutable latest marker. Keep raw Hydra overrides
# supported for callers that do not use this optional environment variable.
resume_checkpoint_arg=""
if [[ -n "$RESUME_FROM_PATH" ]]; then
  if [[ "$RESUME_MODE" != "resume_path" ]]; then
    echo "RESUME_FROM_PATH requires RESUME_MODE=resume_path; clear it when using auto/disable." >&2
    exit 2
  fi
  resume_checkpoint_arg="$(python - "$RESUME_FROM_PATH" "$N_GPUS" <<'PY'
import json
from pathlib import Path
import re
import sys

try:
    checkpoint = Path(sys.argv[1]).expanduser().resolve()
    devices = int(sys.argv[2])
    if not re.fullmatch(r"global_step_[1-9][0-9]*", checkpoint.name):
        raise ValueError("source must name an explicit global_step_N directory")
    actor = checkpoint / "actor"
    world_size = json.loads((actor / "fsdp_config.json").read_text())["world_size"]
    if world_size != devices:
        raise ValueError(f"checkpoint world_size={world_size} differs from visible devices={devices}")
    required = [checkpoint / "data.pt"]
    required += [actor / f"{kind}_world_size_{devices}_rank_{rank}.pt"
                 for kind in ("model", "optim", "extra_state") for rank in range(devices)]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError(f"missing or empty checkpoint state: {', '.join(missing)}")
except (OSError, ValueError, KeyError) as exc:
    raise SystemExit(f"Invalid resume checkpoint: {exc}") from None

# JSON quoting preserves spaces and Hydra-sensitive characters in the path.
print(json.dumps(str(checkpoint)))
PY
)"
  echo "Resume checkpoint: source=$RESUME_FROM_PATH world_size=$N_GPUS mode=$RESUME_MODE"
fi

case "$EVAL_MODE" in
  grounded|text) ;;
  *)
    echo "EVAL_MODE must be 'grounded' or 'text'; got: $EVAL_MODE" >&2
    exit 2
    ;;
esac

case "${AUTO_EXPORT_MODEL,,}" in
  1|true|yes|on)
    auto_export_model=true
    ;;
  0|false|no|off)
    auto_export_model=false
    ;;
  *)
    echo "Invalid AUTO_EXPORT_MODEL=$AUTO_EXPORT_MODEL (expected true or false)" >&2
    exit 2
    ;;
esac

case "${USE_FUSED_KERNELS,,}" in
  1|true|yes|on)
    use_fused_kernels=true
    ;;
  0|false|no|off)
    use_fused_kernels=false
    ;;
  *)
    echo "Invalid USE_FUSED_KERNELS=$USE_FUSED_KERNELS (expected true or false)" >&2
    exit 2
    ;;
esac

case "${PERSIST_COMPILE_CACHE,,}" in
  1|true|yes|on)
    persist_compile_cache=true
    ;;
  0|false|no|off)
    persist_compile_cache=false
    ;;
  *)
    echo "Invalid PERSIST_COMPILE_CACHE=$PERSIST_COMPILE_CACHE (expected true or false)" >&2
    exit 2
    ;;
esac

if ! [[ "$COMPILE_CACHE_SNAPSHOT_DELAY" =~ ^[0-9]+$ ]]; then
  echo "Invalid COMPILE_CACHE_SNAPSHOT_DELAY=$COMPILE_CACHE_SNAPSHOT_DELAY (expected non-negative seconds)" >&2
  exit 2
fi

case "${MOBILEGYM_CONFIG_ONLY_VALUE,,}" in
  1|true|yes|on)
    echo "Model profile: path=$MODEL_PATH type=$MODEL_TYPE slug=$MODEL_SLUG"
    echo "Training devices: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES n_gpus_per_node=$N_GPUS"
    echo "Training namespace: experiment=$EXPERIMENT_NAME run=$RUN_NAME"
    echo "Compile cache: persistent=$COMPILE_CACHE_ROOT local=$COMPILE_CACHE_LOCAL_ROOT"
    echo "MobileGym splits: train=$TRAIN_SPLIT val=$VAL_SPLIT mode=$EVAL_MODE"
    echo "Parameter update: mode=full actor_model_dtype=$ACTOR_MODEL_DTYPE fsdp_compute_dtype=bfloat16"
    echo "Optimization: lr=$LEARNING_RATE warmup_ratio=$LR_WARMUP_STEPS_RATIO weight_decay=$WEIGHT_DECAY train_batch=$TRAIN_BATCH_SIZE mini_batch=$PPO_MINI_BATCH_SIZE clip_low=$PPO_CLIP_RATIO_LOW clip_high=$PPO_CLIP_RATIO_HIGH response_length=$MAX_RESPONSE_LENGTH"
    echo "Sampling: train_sample_n=$SAMPLE_N val_sample_n=$VAL_SAMPLE_N train_temperature=$TRAIN_TEMPERATURE val_temperature=$VAL_TEMPERATURE"
    echo "Schedule: epochs=$EPOCHS total_training_steps=${TOTAL_TRAINING_STEPS:-auto} save_freq=$SAVE_FREQ max_actor_ckpts=$MAX_ACTOR_CKPTS resume=$RESUME_MODE auto_export=$auto_export_model"
    exit 0
    ;;
  0|false|no|off) ;;
  *)
    echo "Invalid MOBILEGYM_CONFIG_ONLY=$MOBILEGYM_CONFIG_ONLY_VALUE (expected true or false)" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$HYDRA_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
# PPU's Triton backend resolves llvm-irformatter and the CUDA-compatible
# toolchain through these variables.  DSW images put the SDK here but do not
# always export the variables into newly spawned Ray/vLLM workers.
if [[ -z "${PPU_SDK:-}" && -d /usr/local/PPU_SDK ]]; then
  export PPU_SDK=/usr/local/PPU_SDK
fi
if [[ -n "${PPU_SDK:-}" && -d "$PPU_SDK/CUDA_SDK" ]]; then
  export CUDA_PATH="${CUDA_PATH:-$PPU_SDK/CUDA_SDK}"
  export CUDA_HOME="${CUDA_HOME:-$CUDA_PATH}"
fi
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export RLLM_GATEWAY_LOG_LEVEL="${RLLM_GATEWAY_LOG_LEVEL:-warning}"
export MOBILE_GYM_TO_THREAD_WORKERS="${MOBILE_GYM_TO_THREAD_WORKERS:-256}"
# Screenshot decoding happens in many threads.  Trim unused glibc arenas once
# the previous global-step batch is dropped; otherwise Ray may kill the
# workflow worker at its 95% host-memory threshold before the cgroup reports an
# OOM.  Do not cap MALLOC_ARENA_MAX: 64-way GUI/image processing needs the
# allocator concurrency.
export RLLM_TRIM_HOST_MEMORY="${RLLM_TRIM_HOST_MEMORY:-1}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

compile_cache_snapshot_pid=""
if [[ "$persist_compile_cache" == true ]]; then
  if [[ -z "${COMPILE_CACHE_NAMESPACE:-}" ]]; then
    COMPILE_CACHE_NAMESPACE="$(python - <<'PY'
import importlib.metadata
import platform
import re


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


raw = "_".join(
    [
        "v1",
        f"py{platform.python_version()}",
        f"torch{package_version('torch')}",
        f"triton{package_version('triton')}",
        f"vllm{package_version('vllm')}",
    ]
)
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", raw))
PY
)"
  fi

  compile_cache_persist_dir="$COMPILE_CACHE_ROOT/$COMPILE_CACHE_NAMESPACE"
  compile_cache_local_dir="$COMPILE_CACHE_LOCAL_ROOT/$COMPILE_CACHE_NAMESPACE"
  # Always keep the active caches on node-local storage.  Besides being faster
  # than CFS for compiler metadata, this also prevents an inherited
  # TRITON_HOME=/root from accidentally making the snapshot cover all of /root.
  export TRITON_HOME="$compile_cache_local_dir/triton_home"
  export TORCHINDUCTOR_CACHE_DIR="$compile_cache_local_dir/torchinductor"
  export VLLM_CACHE_ROOT="$compile_cache_local_dir/vllm"

  compile_cache_names=(triton_home torchinductor vllm)
  compile_cache_local_paths=("$TRITON_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT")

  restore_compile_cache() {
    local index source_dir target_dir
    for index in "${!compile_cache_names[@]}"; do
      source_dir="$compile_cache_persist_dir/${compile_cache_names[$index]}"
      target_dir="${compile_cache_local_paths[$index]}"
      mkdir -p "$target_dir"
      if [[ -d "$source_dir" && "$source_dir" != "$target_dir" ]]; then
        cp -a "$source_dir/." "$target_dir/"
      fi
    done
  }

  snapshot_compile_cache() {
    local index source_dir target_dir snapshot_status=0
    mkdir -p "$compile_cache_persist_dir"
    for index in "${!compile_cache_names[@]}"; do
      source_dir="${compile_cache_local_paths[$index]}"
      target_dir="$compile_cache_persist_dir/${compile_cache_names[$index]}"
      mkdir -p "$target_dir"
      if [[ -d "$source_dir" && "$source_dir" != "$target_dir" ]]; then
        cp -a "$source_dir/." "$target_dir/" || snapshot_status=$?
      fi
    done
    return "$snapshot_status"
  }

  cleanup_compile_cache() {
    local exit_status=$?
    trap - EXIT
    if [[ -n "$compile_cache_snapshot_pid" ]]; then
      kill "$compile_cache_snapshot_pid" 2>/dev/null || true
      wait "$compile_cache_snapshot_pid" 2>/dev/null || true
    fi
    if snapshot_compile_cache; then
      echo "Saved compile cache: $compile_cache_persist_dir"
    else
      echo "Warning: failed to save all compile-cache files to $compile_cache_persist_dir" >&2
    fi
    exit "$exit_status"
  }

  restore_compile_cache
  echo "Compile cache: local=$compile_cache_local_dir persistent=$compile_cache_persist_dir"
  trap cleanup_compile_cache EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT

  if (( COMPILE_CACHE_SNAPSHOT_DELAY > 0 )); then
    (
      trap - EXIT
      sleep "$COMPILE_CACHE_SNAPSHOT_DELAY"
      if snapshot_compile_cache; then
        echo "Saved warm compile-cache snapshot: $compile_cache_persist_dir"
      else
        echo "Warning: warm compile-cache snapshot was incomplete" >&2
      fi
    ) &
    compile_cache_snapshot_pid=$!
  fi
fi

extra_args=()
if [[ -n "$resume_checkpoint_arg" ]]; then
  extra_args+=("trainer.resume_from_path=$resume_checkpoint_arg")
fi
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  extra_args+=("trainer.total_training_steps=$TOTAL_TRAINING_STEPS")
fi
if [[ -n "${MAX_ENV_STEPS:-}" ]]; then
  extra_args+=("+max_steps=$MAX_ENV_STEPS")
fi

VERL_SOURCE="$(python -c 'import pathlib, verl; print(pathlib.Path(verl.__file__).resolve())')"
case "$VERL_SOURCE" in
  "$LOCAL_VERL_ROOT"/*) ;;
  *)
    echo "Expected local VERL under $LOCAL_VERL_ROOT, resolved $VERL_SOURCE" >&2
    exit 2
    ;;
esac
echo "Using local VERL: $VERL_SOURCE"
echo "Model profile: path=$MODEL_PATH type=$MODEL_TYPE slug=$MODEL_SLUG"
echo "Training devices: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES n_gpus_per_node=$N_GPUS"
echo "Actor base dtype: $ACTOR_MODEL_DTYPE"
echo "Chunked LM-head: $use_fused_kernels (torch backend)"
echo "Parameter update: mode=full"
echo "Optimization: lr=$LEARNING_RATE warmup_ratio=$LR_WARMUP_STEPS_RATIO weight_decay=$WEIGHT_DECAY train_batch=$TRAIN_BATCH_SIZE mini_batch=$PPO_MINI_BATCH_SIZE clip_low=$PPO_CLIP_RATIO_LOW clip_high=$PPO_CLIP_RATIO_HIGH response_length=$MAX_RESPONSE_LENGTH"
echo "Sampling: train_sample_n=$SAMPLE_N val_sample_n=$VAL_SAMPLE_N train_temperature=$TRAIN_TEMPERATURE val_temperature=$VAL_TEMPERATURE"
echo "Schedule: epochs=$EPOCHS total_training_steps=${TOTAL_TRAINING_STEPS:-auto} save_freq=$SAVE_FREQ max_actor_ckpts=$MAX_ACTOR_CKPTS resume=$RESUME_MODE auto_export=$auto_export_model"
echo "MobileGym answer mode: $EVAL_MODE"
echo "MobileGym splits: train=$TRAIN_SPLIT val=$VAL_SPLIT"

# Actor rescoring uses the full temperature-scaled distribution.  The rollout
# therefore keeps top-p untruncated so its processed logprobs remain on-policy.
python cookbooks/mobilegym/train.py \
  hydra.run.dir="$HYDRA_OUTPUT_DIR" \
  rllm/backend=verl \
  trainer.project_name=mobilegym \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='[console]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPTS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$EPOCHS" \
  trainer.val_before_train=false \
  trainer.val_only=false \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.use_legacy_worker_impl=disable \
  trainer.balance_batch=true \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +model.name="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.enable_activation_offload=false \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.use_fused_kernels="$use_fused_kernels" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_model_len=40960 \
  actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  +actor_rollout_ref.rollout.limit_images=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enable_prefix_caching=true \
  actor_rollout_ref.rollout.disable_log_stats=false \
  actor_rollout_ref.rollout.temperature="$TRAIN_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.n="$GROUP_SIZE" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype="$ACTOR_MODEL_DTYPE" \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  actor_rollout_ref.actor.optim.weight_decay="$WEIGHT_DECAY" \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.clip_ratio_low="$PPO_CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$PPO_CLIP_RATIO_HIGH" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=true \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=32768 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.return_multi_modal_inputs=true \
  data.dataloader_num_workers=0 \
  data.shuffle=true \
  data.seed="$DATA_SEED" \
  rllm.workflow.n_parallel_tasks="$ENV_POOL_SIZE" \
  rllm.workflow.retry_limit=3 \
  rllm.workflow.raise_on_error=true \
  rllm.gateway.delete_session_after_use=true \
  rllm.compact_filtering.enable=true \
  rllm.compact_filtering.mask_error=true \
  rllm.disable_thinking=true \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_ctrl.kl_coef=0 \
  rllm.algorithm.adv_estimator=grpo \
  +suite=null \
  +run_name="$RUN_NAME" \
  +runs_dir="$LOG_ROOT" \
  +split="$TRAIN_SPLIT" \
  +sample_n="$SAMPLE_N" \
  +task_seed="$TRAIN_SAMPLE_SEED" \
  +shared_val=false \
  +val_split="$VAL_SPLIT" \
  +val_sample_n="$VAL_SAMPLE_N" \
  +val_task_seed="$VAL_SAMPLE_SEED" \
  +max_val_tasks=32 \
  +agent_temperature="$TRAIN_TEMPERATURE" \
  +agent_top_p=1.0 \
  +agent_max_tokens="$MAX_RESPONSE_LENGTH" \
  +agent_val_temperature="$VAL_TEMPERATURE" \
  +agent_val_top_p=0.95 \
  +judge_mode=auto \
  +eval_mode="$EVAL_MODE" \
  +save_trajectory=true \
  +monitor=true \
  +env_pool_size="$ENV_POOL_SIZE" \
  +env_num_browsers="$ENV_NUM_BROWSERS" \
  "${extra_args[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG_DIR/terminal.log"

if [[ "$auto_export_model" == true ]]; then
  python cookbooks/mobilegym/export_latest_checkpoint.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$MODEL_EXPORT_DIR" \
    --device "$EXPORT_DEVICE" \
    --max-shard-size "$EXPORT_MAX_SHARD_SIZE" \
    2>&1 | tee -a "$LOG_DIR/export.log"
fi
