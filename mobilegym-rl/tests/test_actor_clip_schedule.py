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

"""CPU checks for optional outer-step clipping, including real GSPO loss bounds."""
import ast
from dataclasses import dataclass, FrozenInstanceError
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rllm.experimental.clip_schedule import scheduled_actor_config, validate_clip_schedule

SCHEDULE = {'switch_step': 9, 'clip_ratio_low': .002, 'clip_ratio_high': .0028}


@dataclass(frozen=True)
class Config:
    clip_ratio_low: float = .2
    clip_ratio_high: float = .28
    normalization: str = 'unchanged'


def test_boundary_resume_and_original_config_preserved():
    cfg = Config()
    schedule = validate_clip_schedule(SCHEDULE, 16)
    assert scheduled_actor_config(cfg, None, None) is cfg
    for step in (1, 8, 9, 12, 16, 8):
        current = scheduled_actor_config(cfg, schedule, step)
        assert (current.clip_ratio_low, current.clip_ratio_high) == ((.2, .28) if step < 9 else (.002, .0028))
        assert current.normalization == cfg.normalization
    assert (cfg.clip_ratio_low, cfg.clip_ratio_high) == (.2, .28)
    with pytest.raises(FrozenInstanceError):
        cfg.clip_ratio_low = .1


@pytest.mark.parametrize('change', [
    {'switch_step': 1}, {'switch_step': True}, {'switch_step': 17},
    {'clip_ratio_low': 1}, {'clip_ratio_low': -1}, {'clip_ratio_high': 0},
    {'clip_ratio_low': float('nan')}, {'clip_ratio_high': float('inf')},
    {'clip_ratio_low': True}, {'unknown': 1},
])
def test_invalid_schedule_rejected_before_worker_start(change):
    with pytest.raises(ValueError):
        validate_clip_schedule({**SCHEDULE, **change}, 16)


@pytest.mark.parametrize('step', [None, 0, -1, True, 9.0])
def test_missing_or_invalid_distributed_step_fails_instead_of_silently_staying_wide(step):
    with pytest.raises(ValueError, match='actor_global_step'):
        scheduled_actor_config(Config(), SCHEDULE, step)


def _extract(name):
    source = Path(__file__).resolve().parents[1] / 'rllm/experimental/verl/verl_backend.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return node


def test_actual_sender_and_cloudpickled_worker_loss_use_the_same_outer_step(monkeypatch):
    import cloudpickle
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import reduce_metrics
    from verl.workers.config import FSDPActorConfig
    from verl.workers.utils import losses

    cfg = FSDPActorConfig(strategy='fsdp', rollout_n=8, use_dynamic_bsz=True,
                          clip_ratio_low=.2, clip_ratio_high=.28)
    node = _extract('CustomPPOLoss')
    ns = {'__name__': 'isolated_clip_worker'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_CustomPPOLoss', 'exec'), ns)
    loss_fn = cloudpickle.loads(cloudpickle.dumps(ns['CustomPPOLoss'](cfg, SCHEDULE)))
    observed = []

    def spy(config, model_output, data, dp_group):
        observed.append((config.clip_ratio_low, config.clip_ratio_high, tu.get(data, 'actor_global_step')))
        assert config.loss_agg_mode == cfg.loss_agg_mode
        return 0., {}

    monkeypatch.setattr(losses, 'ppo_loss', spy)
    backend = _extract('VerlBackend')
    method = next(n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == '_update_actor_with_loss_routing')
    scope = {'DataProto': object, 'TrainerState': object, 'Any': Any, 'tu': tu,
             'left_right_2_no_padding': lambda x: x, 'reduce_metrics': reduce_metrics}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual_actor_sender', 'exec'), scope)

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    def worker(data):
        # Slicing mimics distributed/microbatch transport of scalar metadata.
        _, metrics = loss_fn(None, data[:1])
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={'metrics': metrics})

    root = AttrDict(actor_clip_schedule=SCHEDULE, actor_rollout_ref=SimpleNamespace(actor=cfg, rollout=SimpleNamespace(n=8)))
    owner = SimpleNamespace(config=root, algorithm_config=None,
                            actor_rollout_wg=SimpleNamespace(update_actor=worker))
    batch = SimpleNamespace(non_tensor_batch={}, to_tensordict=lambda: TensorDict({}, batch_size=[2]))
    for step in (8, 9, 12, 16):
        state = SimpleNamespace(global_step=step, metrics={})
        scope['_update_actor_with_loss_routing'](owner, batch, state)
        assert state.metrics['actor/clip_ratio_low'] == (.2 if step < 9 else .002)
    assert observed == [(.2, .28, 8), (.002, .0028, 9), (.002, .0028, 12), (.002, .0028, 16)]
    assert (loss_fn.config.clip_ratio_low, loss_fn.config.clip_ratio_high) == (.2, .28)


@pytest.mark.parametrize('ratio,advantage,expected_late', [(1.01, 1., -1.0028), (.99, -1., .998)])
def test_actual_gspo_objective_uses_scheduled_upper_and_lower_bounds(ratio, advantage, expected_late):
    import torch
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.workers.config import FSDPActorConfig

    cfg = FSDPActorConfig(strategy='fsdp', rollout_n=8, use_dynamic_bsz=True,
                          clip_ratio_low=.2, clip_ratio_high=.28)
    for step, expected in ((8, -advantage * ratio), (9, expected_late), (16, expected_late)):
        log_prob = torch.tensor([[math.log(ratio)]], requires_grad=True)
        loss, metrics = compute_policy_loss_gspo(torch.zeros_like(log_prob), log_prob,
            torch.full_like(log_prob, advantage), torch.ones_like(log_prob),
            config=scheduled_actor_config(cfg, SCHEDULE, step))
        assert loss.item() == pytest.approx(expected)
        loss.backward()
        assert metrics['actor/pg_clipfrac'] == (0 if step < 9 else 1)
        assert (log_prob.grad.abs().item() > 0) is (step < 9)
