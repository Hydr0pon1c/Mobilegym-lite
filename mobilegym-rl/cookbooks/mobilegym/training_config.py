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

"""Pass JSON environment defaults and Hydra overrides to the existing RL entry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def launch_command(path, inherited, extra_args=()):
    config = json.loads(Path(path).read_text())
    if set(config) != {"environment", "overrides"}:
        raise ValueError("Training config must contain environment and overrides")
    env = {**config["environment"], **inherited}
    env.pop("TRAIN_CONFIG", None)  # Re-enter the existing shell only once.
    command = ["bash", str(Path(__file__).with_name("run_qwen3_vl_4b_env_effect_v3_full.sh"))]
    # JSON quoting also protects URLs containing '=' from Hydra's parser.
    command += [f"++{key}={json.dumps(value)}" for key, value in config["overrides"].items()]
    return env, command + list(extra_args)


if __name__ == "__main__":
    if os.environ.get("MOBILEGYM_PREFLIGHT_ONLY", "0").lower() not in ("0", "false", ""):
        raise SystemExit("Training preflight was removed; use MOBILEGYM_CONFIG_ONLY=1 to inspect settings")
    env, command = launch_command(os.environ["TRAIN_CONFIG"], dict(os.environ), sys.argv[1:])
    print("Configured training overrides:", " ".join(command[2:]), flush=True)
    os.execvpe(command[0], command, env)
