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

# Compatibility entry: shared optimization/model defaults live in the launcher.
# Keep this profile's fixed 96-browser defaults when batch size is overridden.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
  ENV_POOL_SIZE="${ENV_POOL_SIZE:-96}" \
  ENV_NUM_BROWSERS="${ENV_NUM_BROWSERS:-96}" \
  bash "$SCRIPT_DIR/train_mobilegym_gspo_full.sh" "$@"
