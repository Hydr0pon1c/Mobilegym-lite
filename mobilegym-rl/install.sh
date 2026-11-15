#!/usr/bin/env bash
# Copyright 2026 OPPO
#
# Modified by OPPO for this release.
# OPPO copyright applies only to the modifications.
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
cd "$SCRIPT_DIR"
# Hardware packages must already match the selected accelerator image.
python - <<'CHECK'
import importlib.util
import sys
missing = [name for name in ("torch", "vllm", "torchvision", "flash_attn")
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("Install hardware-compatible packages first: " + ", ".join(missing))
CHECK
command -v uv >/dev/null || { echo "Install uv before running this installer." >&2; exit 2; }
python -m pip install -r requirements-runtime.txt
python -m pip install --no-deps -e ./verl
uv pip install --python "$(command -v python)" --no-deps -e ./rllm-model-gateway
python -m pip install --no-deps -e .
python -m playwright install chromium
