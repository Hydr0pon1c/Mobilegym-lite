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

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


RL_ROOT = Path(__file__).resolve().parents[1]
COOKBOOK = RL_ROOT / "cookbooks" / "mobilegym"


def _model_dir(tmp_path: Path, name: str, model_type: str) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )
    return model_dir


def _resume_checkpoint(tmp_path: Path, world_size: int = 4) -> Path:
    checkpoint = tmp_path / "source [branch]" / "global_step_12"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text(json.dumps({"world_size": world_size}))
    (checkpoint / "data.pt").write_bytes(b"dataloader state fixture")
    for kind in ("model", "optim", "extra_state"):
        for rank in range(world_size):
            (actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"state fixture")
    return checkpoint


def _run_config_only(script: str, model_dir: Path, **overrides: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "MOBILEGYM_CONFIG_ONLY": "1",
            "MODEL_PATH": str(model_dir),
            "RUN_NAME": "config-test",
            "EVAL_MODE": "text",
            "EVAL_CONFIG": "",
            "EVAL_COMPLETENESS_MODES": "off",
        }
    )
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(COOKBOOK / script)],
        cwd=RL_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_generic_eval_entrypoint_derives_served_model_name(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")

    output = _run_config_only("eval_mobilegym_4ppu_3runs.sh", model_dir)

    assert "served_name=qwen3-vl-4b-instruct" in output
    assert "split=test mode=text sample_n=1 seeds=42 43 44 repeat_n=1 temperature=0" in output
    assert "Eval resource conditions (sequential, shared model server): off" in output


def test_eval_rejects_retired_recovery_probe_before_startup(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "model", "qwen3_vl")
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_config_only("eval_mobilegym_4ppu_3runs.sh", model_dir,
                         EVAL_RECOVERY_PLAN="closed-probe.json")
    assert error.value.returncode == 2
    assert "EVAL_RECOVERY_PLAN is retired" in error.value.stderr


def test_eval_resource_panel_reuses_model_without_multiplying_concurrency(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "global_step_16", "qwen3_vl")
    output = _run_config_only(
        "eval_mobilegym_4ppu_3runs.sh", model_dir,
        EVAL_COMPLETENESS_MODES="resources_full resources_0 resources_1 resources_2",
        EVAL_SAMPLE_SEEDS="42", EVAL_PARALLEL="16",
        EVAL_PROCESSES="2", EVAL_BROWSERS="2",
        EVAL_ENV_URL="https://127.0.0.1:4180/?benchmark_time_ms=1789099200000",
    )
    assert "seeds=42 repeat_n=1 temperature=0" in output
    assert "sequential, shared model server): resources_full resources_0 resources_1 resources_2" in output
    assert "per_seed=16 total=16 processes_per_seed=2 browsers_per_seed=2" in output
    assert "benchmark_time_ms=1789099200000" in output


def test_eval_lite_is_opt_in_and_can_share_a_full_diagnostic_panel(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "model", "qwen3_vl")
    output = _run_config_only("eval_mobilegym_4ppu_3runs.sh", model_dir,
                              EVAL_COMPLETENESS_MODES="off lite_v1")
    assert "sequential, shared model server): off lite_v1" in output


def test_eval_selective_is_opt_in_and_keeps_full_in_the_panel(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "model", "qwen3_vl")
    output = _run_config_only("eval_mobilegym_4ppu_3runs.sh", model_dir,
                              EVAL_COMPLETENESS_MODES="off feature0_selective_v1")
    assert "sequential, shared model server): off feature0_selective_v1" in output


@pytest.mark.parametrize("modes", ["resources_typo", "resources_0 resources_0", "   "])
def test_eval_rejects_invalid_or_duplicate_resource_conditions(tmp_path: Path, modes: str):
    model_dir = _model_dir(tmp_path, "model", "qwen3_vl")
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_config_only("eval_mobilegym_4ppu_3runs.sh", model_dir, EVAL_COMPLETENESS_MODES=modes)
    assert error.value.returncode == 2


def test_generic_full_training_defaults_to_v3(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")

    output = _run_config_only("train_mobilegym_gspo_full.sh", model_dir)

    assert "type=qwen3_vl slug=qwen3-vl-4b-instruct" in output
    assert "experiment=mobilegym-qwen3-vl-4b-instruct-gspo-full" in output
    assert "cache/qwen3_vl_4b_instruct" in output
    assert "MobileGym splits: train=env_effect_v3_train val=env_effect_v3_test mode=text" in output
    assert "Parameter update: mode=full actor_model_dtype=fp32" in output
    assert "Schedule: epochs=1 total_training_steps=auto" in output


def test_qwen3_vl_full_parameter_profile_matches_paper_scale(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")

    output = _run_config_only("train_qwen3_vl_4b_gspo_full.sh", model_dir)

    assert "experiment=mobilegym-qwen3-vl-4b-instruct-gspo-full" in output
    assert "MobileGym splits: train=env_effect_v3_train val=env_effect_v3_test mode=text" in output
    assert "Parameter update: mode=full actor_model_dtype=fp32" in output
    assert (
        "Optimization: lr=1e-6 warmup_ratio=0.0 weight_decay=0.01 "
        "train_batch=12 mini_batch=12 clip_low=0.2 clip_high=0.28 "
        "response_length=1024"
    ) in output
    assert (
        "Sampling: train_sample_n=1 val_sample_n=1 "
        "train_temperature=0.7 val_temperature=0"
    ) in output


def test_env_effect_v3_full_platform_entrypoint_is_configurable(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")

    output = _run_config_only(
        "run_qwen3_vl_4b_env_effect_v3_full.sh",
        model_dir,
        MOBILEGYM_ROOT=str(RL_ROOT.parent),
        EPOCHS="3",
    )

    assert "Parameter update: mode=full actor_model_dtype=fp32" in output
    assert "MobileGym splits: train=env_effect_v3_train val=env_effect_v3_test mode=text" in output
    assert "Sampling: train_sample_n=1 val_sample_n=1" in output
    assert "Schedule: epochs=3 total_training_steps=auto" in output


def test_env_effect_v3_full_platform_entrypoint_honors_split_and_sample_overrides(
    tmp_path: Path,
):
    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")

    output = _run_config_only(
        "run_qwen3_vl_4b_env_effect_v3_full.sh",
        model_dir,
        MOBILEGYM_ROOT=str(RL_ROOT.parent),
        TRAIN_SPLIT="env_effect_v3_shared",
        VAL_SPLIT="env_effect_v3_ood_test",
        SAMPLE_N="2",
    )

    assert (
        "MobileGym splits: train=env_effect_v3_shared "
        "val=env_effect_v3_ood_test mode=text"
    ) in output
    assert "Sampling: train_sample_n=2 val_sample_n=1" in output


def test_explicit_checkpoint_branch_is_checked_without_writing_artifacts(tmp_path: Path):
    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")
    checkpoint = _resume_checkpoint(tmp_path)
    artifact_root = tmp_path / "not-created"
    output = _run_config_only(
        "run_qwen3_vl_4b_env_effect_v3_full.sh", model_dir,
        MOBILEGYM_ROOT=str(RL_ROOT.parent), CUDA_VISIBLE_DEVICES="0,1,2,3",
        RESUME_MODE="resume_path", RESUME_FROM_PATH=str(checkpoint),
        TRAIN_ARTIFACT_ROOT=str(artifact_root),
        PPO_CLIP_RATIO_LOW="0.002", PPO_CLIP_RATIO_HIGH="0.0028",
    )
    assert f"Resume checkpoint: source={checkpoint} world_size=4 mode=resume_path" in output
    assert "clip_low=0.002 clip_high=0.0028" in output
    assert not artifact_root.exists()
    assert (checkpoint / "data.pt").read_bytes() == b"dataloader state fixture"


@pytest.mark.parametrize("fault", ["world_size", "data", "optimizer", "mode"])
def test_explicit_checkpoint_branch_rejects_incomplete_or_ignored_restore(tmp_path: Path, fault: str):
    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")
    checkpoint = _resume_checkpoint(tmp_path)
    devices, mode = "0,1,2,3", "resume_path"
    if fault == "world_size":
        devices = "0,1"
    elif fault == "data":
        (checkpoint / "data.pt").unlink()
    elif fault == "optimizer":
        (checkpoint / "actor/optim_world_size_4_rank_3.pt").unlink()
    else:
        mode = "auto"
    artifact_root = tmp_path / "not-created"
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_config_only(
            "train_mobilegym_gspo_full.sh", model_dir,
            CUDA_VISIBLE_DEVICES=devices, RESUME_MODE=mode, RESUME_FROM_PATH=str(checkpoint),
            TRAIN_ARTIFACT_ROOT=str(artifact_root),
        )
    assert "RESUME_FROM_PATH requires" in error.value.stderr or "Invalid resume checkpoint" in error.value.stderr
    assert not artifact_root.exists()


def test_checkpoint_branch_path_reaches_training_hydra_arguments(tmp_path: Path):
    from hydra.core.override_parser.overrides_parser import OverridesParser

    model_dir = _model_dir(tmp_path, "Qwen3-VL-4B-Instruct", "qwen3_vl")
    checkpoint = _resume_checkpoint(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Intercept only the final model entry and the source-path probe; execute
    # the actual shell validation with Python. All artifacts stay in tmp_path.
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1] == 'cookbooks/mobilegym/train.py':\n"
        "    pathlib.Path(os.environ['MOBILEGYM_TEST_CAPTURE_ARGS']).write_text(json.dumps(sys.argv[2:]))\n"
        "elif sys.argv[1] == '-c' and 'import pathlib, verl' in sys.argv[2]:\n"
        f"    print({str(RL_ROOT / 'verl/verl/__init__.py')!r})\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    capture = tmp_path / "arguments.json"
    _run_config_only(
        "train_mobilegym_gspo_full.sh", model_dir,
        MOBILEGYM_CONFIG_ONLY="0", CUDA_VISIBLE_DEVICES="0,1,2,3",
        RESUME_MODE="resume_path", RESUME_FROM_PATH=str(checkpoint),
        AUTO_EXPORT_MODEL="false", PERSIST_COMPILE_CACHE="false",
        TRAIN_ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        MOBILEGYM_TEST_CAPTURE_ARGS=str(capture), PATH=f"{bin_dir}:{os.environ['PATH']}",
        PPO_CLIP_RATIO_LOW="0.002", PPO_CLIP_RATIO_HIGH="0.0028",
    )
    overrides = OverridesParser.create().parse_overrides(json.loads(capture.read_text()))
    values = {item.key_or_group: item.value() for item in overrides}
    assert values["trainer.resume_mode"] == "resume_path"
    assert values["trainer.resume_from_path"] == str(checkpoint.resolve())
    assert values["trainer.n_gpus_per_node"] == 4
    assert values["actor_rollout_ref.actor.clip_ratio_low"] == .002
    assert values["actor_rollout_ref.actor.clip_ratio_high"] == .0028
    assert values["actor_rollout_ref.actor.optim.lr"] == 1e-6
    assert (checkpoint / "data.pt").read_bytes() == b"dataloader state fixture"
