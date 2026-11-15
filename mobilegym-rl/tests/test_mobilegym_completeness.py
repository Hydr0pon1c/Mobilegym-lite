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

"""RL config and rollout routing without a GPU, model API or browser."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cookbooks.mobilegym import mobilegym_flow as flow
from cookbooks.mobilegym.evaluator import mobilegym_evaluator
from cookbooks.mobilegym.training_config import launch_command
from bench_env.runner.base import EpisodeResult, ExecutionResult
from bench_env.task.judge import JudgeResult
from bench_env.feature_completeness import MIX_MODE, LAYOUT_MODE, SELECTIVE_MODE, PLUS_MODE, feature_plan
from rllm.experimental.eval.types import AgentConfig, Task


@pytest.mark.parametrize("reward_mode", ["legacy", "binary_native"])
@pytest.mark.parametrize("condition", ["resources_0", "feature0", "feature1", "lite_v1", MIX_MODE, LAYOUT_MODE, SELECTIVE_MODE, PLUS_MODE])
def test_config_roundtrip(monkeypatch, reward_mode, condition):
    cfg = flow.RuntimeConfig(reward_mode=reward_mode, train_completeness=condition)
    monkeypatch.setenv(flow._ENV_KEY, "")
    cfg.to_env_vars()
    restored = flow.RuntimeConfig.from_env()
    assert restored.train_completeness == condition and restored.val_completeness is None
    assert restored.reward_mode == reward_mode


@pytest.mark.parametrize("condition", ["train_completeness", "val_completeness"])
@pytest.mark.parametrize("kwargs", [
    {"reward_mode": "binary_reference"}, {"val_reward_mode": "binary_reference"},
    {"reference_bundle": "unused"}, {"reward_mode": "binary_native", "reference_bundle": "unused"},
])
def test_reference_conditions_stay_separate(condition, kwargs):
    with pytest.raises(ValueError, match="do not support reference"):
        flow.RuntimeConfig(**{condition: "resources_0"}, **kwargs)


@pytest.mark.parametrize("reward_mode,expected_reward", [("legacy", 0.6), ("binary_native", 0.0)])
@pytest.mark.parametrize("condition", ["resources_2", "feature0", "feature1", "lite_v1", MIX_MODE, LAYOUT_MODE, SELECTIVE_MODE, PLUS_MODE])
def test_flow_preserves_partial_reward_and_keeps_validation_full(monkeypatch, reward_mode, expected_reward, condition):
    cfg = flow.RuntimeConfig(reward_mode=reward_mode, train_completeness=condition)
    env = SimpleNamespace(set_current_task=Mock())
    pool = SimpleNamespace(acquire=AsyncMock(return_value=env), release=Mock())
    state = SimpleNamespace(get_pool=AsyncMock(return_value=pool), get_recorder=AsyncMock(return_value=None),
                            next_trial_id=AsyncMock(side_effect=range(10)), run_dir=None)
    task = SimpleNamespace(id="clock.ToggleAlarm_i0", _seed=41, difficulty="L1")
    monkeypatch.setattr(flow, "_state", state)
    monkeypatch.setattr(flow, "get_config", lambda: cfg)
    monkeypatch.setattr(flow, "load_task_by_id", lambda _: task)
    monkeypatch.setattr(flow, "_build_agent", lambda _: object())
    observed = []

    async def run_episode(*args, **kwargs):
        mode = kwargs["completeness"]
        observed.append(mode)
        assert kwargs['completeness_rollout_index'] == (0 if mode == MIX_MODE else None)
        assert kwargs["setup_observer"] is None
        record = {"mode": mode, "effective": True} if mode else None
        return EpisodeResult(task.id, "test", "clock",
            ExecutionResult(1, [], 0, True, False, stop_reason="COMPLETE"),
            judge=JudgeResult(success=False, clean=True, progress=0.75), completeness=record)

    monkeypatch.setattr(flow.BaseRunner, "run_episode", run_episode)

    async def check():
        for mode in ("train", "val"):
            episode = await flow.mobilegym_flow.arun(Task(data={"task_id": task.id, "task_seed": 41}),
                AgentConfig(base_url="http://unused", model="none", session_uid=f"{mode}:0", metadata={"mode": mode}))
            assert episode.artifacts["reward_mode"] == reward_mode
            assert ("completeness" in episode.artifacts) == (mode == "train")
            output = mobilegym_evaluator({}, episode)
            # Preserve verifier partial credit and the existing false-finish discount.
            assert output.reward == pytest.approx(expected_reward)
            assert not output.is_correct
    asyncio.run(check())
    assert observed == [condition, None] and pool.release.call_count == 2


def test_current_preset_uses_defaults_and_a_new_run_name():
    from hydra.core.override_parser.overrides_parser import OverridesParser
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/training/full.json"
    for mode in ("resources_full", "resources_2", "resources_1", "resources_0"):
        env, command = launch_command(path, {"RUN_NAME": "new_resource_run"}, [f"++train_completeness={mode}"])
        overrides = {v.key_or_group: v.value() for v in OverridesParser.create().parse_overrides(command[2:])}
        assert "reward_mode" not in overrides and "val_reward_mode" not in overrides
        cfg = flow.RuntimeConfig(**{k: v for k, v in overrides.items() if k in flow.RuntimeConfig.__dataclass_fields__})
        assert cfg.reward_mode == flow.RuntimeConfig().reward_mode == "legacy"
        assert cfg.train_completeness == mode and cfg.val_completeness is None
        assert cfg.env_url.endswith("?benchmark_time_ms=1789099200000")
        assert env["RUN_NAME"] == "new_resource_run"
        assert env["RESUME_MODE"] == "disable"


@pytest.mark.parametrize("mode", ["feature0", "feature1", "lite_v1", MIX_MODE, LAYOUT_MODE, SELECTIVE_MODE, PLUS_MODE])
def test_feature_preset_is_portable_and_keeps_validation_full(mode):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    root = Path(__file__).resolve().parents[2]
    env, command = launch_command(root / "configs/training/feature0.json",
        {"RUN_NAME": f"qwen3vl_4b_v3_{mode}_192_v1_e1"}, [f"++train_completeness={mode}"])
    overrides = {v.key_or_group: v.value() for v in OverridesParser.create().parse_overrides(command[2:])}
    cfg = flow.RuntimeConfig(**{k: v for k, v in overrides.items() if k in flow.RuntimeConfig.__dataclass_fields__})
    assert cfg.train_completeness == mode and cfg.val_completeness is None
    assert cfg.reference_bundle is None and overrides["train_samples"] is None
    assert cfg.env_url.endswith("?benchmark_time_ms=1789099200000")
    assert env["TRAIN_SPLIT"] == "env_effect_v3_train"
    assert env["RUN_NAME"] == f"qwen3vl_4b_v3_{mode}_192_v1_e1"
    assert "MOBILEGYM_STATIC_DIR" not in env and "MODEL_PATH" not in env


def test_mix_retry_keeps_mask_despite_different_recording_trial(monkeypatch):
    cfg = flow.RuntimeConfig(train_completeness=MIX_MODE)
    task = SimpleNamespace(id='notes.PinNote_i0', _seed=41, difficulty='L2')
    env = SimpleNamespace(set_current_task=Mock())
    pool = SimpleNamespace(acquire=AsyncMock(return_value=env), release=Mock())
    state = SimpleNamespace(get_pool=AsyncMock(return_value=pool), get_recorder=AsyncMock(return_value=None),
                            next_trial_id=AsyncMock(side_effect=[0, 8, 9]), run_dir=None)
    monkeypatch.setattr(flow, '_state', state)
    monkeypatch.setattr(flow, 'get_config', lambda: cfg)
    monkeypatch.setattr(flow, 'load_task_by_id', lambda _: task)
    monkeypatch.setattr(flow, '_build_agent', lambda _: object())
    records = []

    async def run_episode(*args, **kwargs):
        record = feature_plan(task, kwargs['completeness'], rollout_index=kwargs['completeness_rollout_index'])
        records.append((kwargs['trial_id'], record))
        error = len(records) == 1
        return EpisodeResult(task.id, 'pin', 'notes',
            ExecutionResult(0, [], 0, not error, False, stop_reason='ERROR' if error else 'COMPLETE',
                            error='temporary failure' if error else None),
            judge=JudgeResult(success=not error, clean=True, progress=int(not error)), completeness=record)

    monkeypatch.setattr(flow.BaseRunner, 'run_episode', run_episode)

    async def check():
        index = feature_plan(task, MIX_MODE)['mixture']['full_slot']
        config = AgentConfig(base_url='http://unused', model='none', session_uid=f'group:{index}', metadata={'mode': 'train'})
        with pytest.raises(RuntimeError, match='Infra error'):
            await flow.mobilegym_flow.arun(Task(data={'task_id': task.id}), config)
        episode = await flow.mobilegym_flow.arun(Task(data={'task_id': task.id}), config)
        assert episode.artifacts['completeness'] == records[0][1] == records[1][1]
        assert records[0][0] == 0 and records[1][0] == 8
        assert not records[1][1]['disabled']
        config.session_uid = 'missing-stable-index'
        with pytest.raises(ValueError, match='session UID'):
            await flow.mobilegym_flow.arun(Task(data={'task_id': task.id}), config)
    asyncio.run(check())
    assert pool.release.call_count == 3


@pytest.mark.parametrize('n', [1, 4, 8, 16])
def test_mix_training_requires_eight_rollouts(n):
    from omegaconf import OmegaConf
    from cookbooks.mobilegym.train import _build_runtime_config
    config = OmegaConf.create({'train_completeness': MIX_MODE, 'actor_rollout_ref': {'rollout': {'n': n}},
                              'data': {}, 'rllm': {'trainer': {'project_name': 'test', 'experiment_name': 'test'}}})
    if n == 8:
        assert _build_runtime_config(config, 96).train_completeness == MIX_MODE
    else:
        with pytest.raises(ValueError, match='requires rollout.n=8'):
            _build_runtime_config(config, 96)


@pytest.mark.parametrize('stop_reason,message,expected', [
    ('ABORT', 'invalid_json', 0.0),
    ('FORMAT_ERROR', None, 0.0),
    ('ABORT', 'Cannot find the remaining setting', 0.4),
    ('COMPLETE', 'Done', 0.4),
])
@pytest.mark.parametrize('goal_success', [False, True])
def test_format_failures_do_not_inherit_partial_credit(stop_reason, message, expected, goal_success):
    from bench_env.env.base import ActionType
    stop_reason = ActionType(stop_reason) if stop_reason != 'FORMAT_ERROR' else stop_reason
    if goal_success and expected:
        expected = 0.5 if stop_reason == ActionType.ABORT else 1.0
    result = EpisodeResult('task', 'test', 'clock',
        ExecutionResult(2, [], 0, True, False, stop_reason=stop_reason, agent_message=message),
        judge=JudgeResult(success=goal_success, clean=True, progress=1.0 if goal_success else 0.5))
    episode = flow._episode_from_runner_result({'task_id': 'task'}, result, None)
    output = mobilegym_evaluator({}, episode)
    assert output.reward == pytest.approx(expected)
    assert output.is_correct == (goal_success and stop_reason == ActionType.COMPLETE)


@pytest.mark.parametrize('as_enum', [False, True])
def test_abort_after_goal_gets_same_discount_in_memory_and_after_json(as_enum):
    from bench_env.env.base import ActionType
    stop = ActionType.ABORT if as_enum else 'ABORT'
    result = EpisodeResult('task', 'test', 'clock',
        ExecutionResult(2, [], 0, True, False, stop_reason=stop, agent_message='giving up'),
        judge=JudgeResult(success=True, clean=True, progress=1.0))
    episode = flow._episode_from_runner_result({'task_id': 'task'}, result, None)
    output = mobilegym_evaluator({}, episode)
    assert output.reward == pytest.approx(0.5)
    assert not output.is_correct
    assert next(s.value for s in output.signals if s.name == 'aborted_after_goal_success') == 1.0
