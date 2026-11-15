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

# Shared environment-effect training entry.
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
if [[ -n "${TRAIN_CONFIG:-}" ]]; then
  exec "$python_bin" "$MOBILEGYM_ROOT/mobilegym-rl/cookbooks/mobilegym/training_config.py" "$@"
fi
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-VL checkpoint directory}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
API_WORKERS="${API_WORKERS:-16}"
GATEWAY_PORT="${GATEWAY_PORT:-4180}"

gateway_script="$MOBILEGYM_ROOT/scripts/server/start_nginx_gateway.sh"
train_script="$MOBILEGYM_ROOT/mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_gspo_full.sh"
config_only="${MOBILEGYM_CONFIG_ONLY:-false}"

for required_file in "$gateway_script" "$train_script"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 2
  fi
done

if [[ "$config_only" != "1" && "$config_only" != "true" ]]; then
  if [[ ! -f "$python_bin" ]]; then
    echo "Required file not found: $python_bin" >&2
    exit 2
  fi
  cd "$MOBILEGYM_ROOT"
  API_WORKERS="$API_WORKERS" PYTHON_BIN="$python_bin" \
    bash "$gateway_script" "$GATEWAY_PORT"
fi

cd "$MOBILEGYM_ROOT/mobilegym-rl"
exec env \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}" \
  MODEL_PATH="$MODEL_PATH" \
  TRAIN_SPLIT="${TRAIN_SPLIT:-env_effect_v3_train}" \
  VAL_SPLIT="${VAL_SPLIT:-env_effect_v3_test}" \
  EVAL_MODE="${EVAL_MODE:-text}" \
  SAMPLE_N="${SAMPLE_N:-1}" \
  VAL_SAMPLE_N="${VAL_SAMPLE_N:-1}" \
  TRAIN_SAMPLE_SEED="${TRAIN_SAMPLE_SEED:-41}" \
  VAL_SAMPLE_SEED="${VAL_SAMPLE_SEED:-42}" \
  DATA_SEED="${DATA_SEED:-42}" \
  EPOCHS="${EPOCHS:-1}" \
  SAVE_FREQ="${SAVE_FREQ:-8}" \
  MAX_ACTOR_CKPTS="${MAX_ACTOR_CKPTS:-null}" \
  TEST_FREQ="${TEST_FREQ:-0}" \
  RESUME_MODE="${RESUME_MODE:-disable}" \
  AUTO_EXPORT_MODEL="${AUTO_EXPORT_MODEL:-true}" \
  RUN_NAME="$RUN_NAME" \
  bash "$train_script" "$@"
