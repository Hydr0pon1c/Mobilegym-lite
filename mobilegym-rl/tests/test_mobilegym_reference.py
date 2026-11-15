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

"""Binary reward and instance binding without a GPU or model API."""
import json
from types import SimpleNamespace

import pytest

from cookbooks.mobilegym import mobilegym_flow as flow
from cookbooks.mobilegym.evaluator import mobilegym_evaluator
from bench_env.reference_runtime import load_reference_bundle, reference_entry


@pytest.fixture
def bundle_path(tmp_path):
    path = tmp_path / 'canonical.json'
    path.write_text(json.dumps(dict(schema_version=2, profile='practical_v1', time_ms=1789099200000,
        tasks=[dict(task_id='clock.ToggleAlarm', task_seed=123, operations=[])])))
    return str(path)


def test_runtime_freezes_time_and_requires_reference_instance(bundle_path):
    options = dict(reference_bundle=bundle_path, eval_mode='text', judge_mode='state')
    cfg = flow.RuntimeConfig(reward_mode='binary_reference', **options)
    assert 'benchmark_time_ms=1789099200000' in cfg.env_url
    with pytest.raises(ValueError, match='conflicts'):
        flow.RuntimeConfig(**options, env_url='https://localhost/?benchmark_time_ms=1')
    with pytest.raises(ValueError, match='needs reference_bundle'):
        flow.RuntimeConfig(reward_mode='binary_reference')
    bundle = load_reference_bundle(bundle_path)
    assert reference_entry(bundle, {'task_id': 'clock.ToggleAlarm', 'task_seed': 123})
    for row in ({'task_id': 'clock.ToggleAlarm', 'task_seed': 41}, {'task_id': 'absent', 'task_seed': 123}):
        with pytest.raises(ValueError, match='instance mismatch'):
            reference_entry(bundle, row)


@pytest.mark.parametrize('mode,success,matched,expected', [
    ('binary_native', True, False, 1), ('binary_native', False, True, 0),
    ('binary_reference', True, True, 1), ('binary_reference', True, False, 0),
    ('binary_reference', False, True, 0), ('binary_reference', False, False, 0),
])
def test_binary_reward_ignores_fractional_progress(mode, success, matched, expected):
    artifacts = dict(reward_mode=mode, success=success, progress=0.75,
                     reference=dict(match=matched, fallback_steps=0, overhead_s=0.1))
    output = mobilegym_evaluator({}, SimpleNamespace(artifacts=artifacts))
    assert output.reward == expected
    assert output.is_correct == success  # Native success stays visible independently.


def test_missing_evidence_is_an_integration_error_not_a_silent_negative():
    with pytest.raises(ValueError, match='Missing reference evidence'):
        mobilegym_evaluator({}, SimpleNamespace(artifacts=dict(reward_mode='binary_reference', success=True)))


def test_native_validation_is_default_for_both_binary_arms(bundle_path):
    for mode in ('binary_native', 'binary_reference'):
        cfg = flow.RuntimeConfig(reward_mode=mode, reference_bundle=bundle_path, eval_mode='text', judge_mode='state')
        assert cfg.val_reward_mode is None
    with pytest.raises(ValueError, match='Validation uses native'):
        flow.RuntimeConfig(reference_bundle=bundle_path, val_reward_mode='binary_reference')
