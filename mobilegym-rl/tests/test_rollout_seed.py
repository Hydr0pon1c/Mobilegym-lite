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

"""Typed rollout configuration must preserve the backend's replica seed offsets."""
import ast
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
from verl.utils.config import omega_conf_to_dataclass


def rollout_config(**kwargs):
    return omega_conf_to_dataclass(OmegaConf.create({
        '_target_': 'verl.workers.config.RolloutConfig', 'name': 'vllm', **kwargs,
    }))


def test_default_seed_preserves_legacy_server_base():
    assert rollout_config().get('seed', 0) == 0


@pytest.mark.parametrize('seed', [1011, 2022, 3033])
def test_configured_seed_reaches_existing_replica_offset(seed):
    config = rollout_config(seed=seed)
    assert config.seed == seed
    source = Path(__file__).resolve().parents[1] / 'verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py'
    expressions = [value for node in ast.walk(ast.parse(source.read_text()))
                   if isinstance(node, ast.Dict)
                   for key, value in zip(node.keys, node.values)
                   if isinstance(key, ast.Constant) and key.value == 'seed'
                   and isinstance(value, ast.BinOp)]
    assert len(expressions) == 1
    expression = compile(ast.Expression(expressions[0]), str(source), 'eval')
    actual = [eval(expression, {'self': SimpleNamespace(config=config, replica_rank=r)})
              for r in range(4)]
    assert actual == [seed, seed + 1, seed + 2, seed + 3]
