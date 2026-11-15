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

"""Public configurations and relocation checks; no services or accelerators."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from cookbooks.mobilegym.training_config import launch_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name,mode", [
    ("full", None), ("feature0", "feature0"),
    ("feature0-selective", "feature0_selective_v1"),
])
def test_public_configs_keep_same_protocol_and_full_validation(name, mode):
    spec = json.loads((ROOT / f"configs/training/{name}.json").read_text())
    assert spec["overrides"]["train_completeness"] == mode
    assert spec["overrides"]["val_completeness"] is None
    assert spec["overrides"]["reference_bundle"] is None
    assert spec["environment"]["TRAIN_SPLIT"] == "env_effect_v3_train"
    assert spec["environment"]["VAL_SPLIT"] == "env_effect_v3_test"
    assert not {"MODEL_PATH", "RUN_NAME", "MOBILEGYM_STATIC_DIR"} & spec["environment"].keys()
    baseline = json.loads((ROOT / "configs/training/full.json").read_text())
    baseline["overrides"]["train_completeness"] = mode
    assert spec == baseline


def test_inherited_environment_and_explicit_overrides_take_precedence():
    spec = ROOT / "configs/training/feature0.json"
    env, command = launch_command(spec, {
        "RUN_NAME": "new_run", "EPOCHS": "2", "TRAIN_CONFIG": str(spec),
    }, ["++train_completeness=null"])
    assert env["EPOCHS"] == "2" and env["RUN_NAME"] == "new_run"
    assert "TRAIN_CONFIG" not in env
    assert command[-1] == "++train_completeness=null"


def test_launcher_relocates_and_runs_from_an_unrelated_working_directory(tmp_path):
    checkout = tmp_path / "relocated checkout"
    paths = [
        "scripts/server/start_nginx_gateway.sh",
        "mobilegym-rl/cookbooks/mobilegym/run_qwen3_vl_4b_env_effect_v3_full.sh",
        "mobilegym-rl/cookbooks/mobilegym/train_qwen3_vl_4b_gspo_full.sh",
        "mobilegym-rl/cookbooks/mobilegym/train_mobilegym_gspo_full.sh",
        "mobilegym-rl/cookbooks/mobilegym/training_config.py",
        "mobilegym-rl/cookbooks/mobilegym/eval_mobilegym_4ppu_3runs.sh",
        "mobilegym-rl/verl/verl/__init__.py",
        "configs/training/feature0.json",
    ]
    for name in paths:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    env = {key: value for key, value in os.environ.items()
           if key not in {"MOBILEGYM_ROOT", "RL_VENV", "EVAL_CONFIG", "EVAL_RECOVERY_PLAN"}}
    env.update({
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "MODEL_PATH": str(tmp_path / "local model"),
        "RUN_NAME": "relocation_check", "MOBILEGYM_CONFIG_ONLY": "1",
        "TRAIN_CONFIG": str(checkout / "configs/training/feature0.json"),
    })
    script = checkout / paths[1]
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    assert '++train_completeness="feature0"' in result.stdout
    assert "train=env_effect_v3_train val=env_effect_v3_test" in result.stdout
    assert str(checkout / "mobilegym-rl/../artifacts/train") in result.stdout
    env.pop("TRAIN_CONFIG")
    result = subprocess.run(["bash", str(checkout / paths[5])], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    assert str(checkout / "dist") in result.stdout
    assert "shared model server): off" in result.stdout
    assert not (checkout / "artifacts").exists()
    assert not (checkout / ".nginx").exists()
