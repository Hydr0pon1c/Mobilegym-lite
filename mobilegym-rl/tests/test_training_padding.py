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

"""CPU regression: DP padding must not become additional GSPO training samples."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.workers.config.actor import ActorConfig

from verl.trainer.ppo import core_algos


# Importing the backend also imports the gateway's provider SDKs. These exact
# method bodies need only CPU tensors and DataProto; no Ray or API clients.
ROOT = Path(__file__).resolve().parents[1]
namespace = dict(np=np, torch=torch, DataProto=DataProto,
                 AdvantageEstimator=core_algos.AdvantageEstimator, core_algos=core_algos)


def load_functions(relative, names, class_name=None):
    path = ROOT / relative
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


load_functions('verl/verl/trainer/ppo/ray_trainer.py', ['compute_advantage'])
load_functions('rllm/experimental/verl/verl_advantage.py',
               ['compute_advantage_verl', '_stepwise_advantage_broadcast'])
load_functions('rllm/experimental/verl/verl_backend.py',
               ['_pad_dataproto_to_world_size'], 'VerlBackend')
compute_advantage_verl = namespace['compute_advantage_verl']
pad_batch = namespace['_pad_dataproto_to_world_size']


@pytest.mark.parametrize('world_size', [2, 4, 16])
def test_padding_has_zero_gradient_after_grpo_broadcast_and_reordering(world_size):
    # Two rollouts of one task, three GUI responses each, unequal rewards.
    batch = DataProto.from_dict(
        tensors={
            'prompts': torch.ones(6, 2, dtype=torch.long),
            'responses': torch.ones(6, 3, dtype=torch.long),
            'response_mask': torch.ones(6, 3),
            'traj_rewards': torch.tensor([[0., 0., x] for x in [1., 1., 1., 0., 0., 0.]]),
            'step_rewards': torch.zeros(6, 3),
        },
        non_tensors={
            'trajectory_ids': np.array(['task_agent'] * 6),
            'episode_ids': np.array(['high'] * 3 + ['low'] * 3),
            'step_nums': np.array([3] * 6),
            'is_last_step': np.array([False, False, True] * 2),
            'is_pad_step': np.zeros(6, dtype=bool),
            'is_valid': np.ones(6, dtype=bool),
        },
    )
    backend = SimpleNamespace(
        _get_dp_world_size=lambda: world_size,
        _remove_padding=lambda b: b.select_idxs(np.where(~b.non_tensor_batch['is_pad_step'])[0]),
    )
    padded = pad_batch(backend, batch)
    # Balancing changes row order; advantage broadcasting must still match episodes.
    padded.reorder(torch.arange(len(padded) - 1, -1, -1))
    cfg = OmegaConf.create({
        'rllm': {'stepwise_advantage': {'mode': 'broadcast', 'normalize_by_steps': False}},
        'algorithm': {'adv_estimator': 'grpo', 'gamma': 1., 'lam': 1.,
                      'norm_adv_by_std_in_grpo': True, 'use_kl_in_reward': False},
        'actor_rollout_ref': {'rollout': {'n': 2}},
    })
    trained, _ = compute_advantage_verl(padded, cfg)
    pad = torch.tensor(trained.non_tensor_batch['is_pad_step'])
    log_prob = torch.zeros_like(trained.batch['advantages'], requires_grad=True)
    loss, _ = compute_policy_loss_gspo(
        old_log_prob=torch.zeros_like(log_prob), log_prob=log_prob,
        advantages=trained.batch['advantages'], response_mask=trained.batch['response_mask'],
        config=ActorConfig(strategy='fsdp', rollout_n=2, use_dynamic_bsz=True,
                           clip_ratio_low=.2, clip_ratio_high=.28),
    )
    loss.backward()
    assert torch.count_nonzero(log_prob.grad[pad]) == 0
    assert torch.count_nonzero(log_prob.grad[~pad]) == 18
    high = torch.tensor(trained.non_tensor_batch['episode_ids'] == 'high') & ~pad
    assert (log_prob.grad[high] < 0).all()
    assert (log_prob.grad[~high & ~pad] > 0).all()
    assert torch.count_nonzero(trained.batch['response_mask'][pad]) == 0
    assert torch.count_nonzero(trained.batch['token_level_rewards'][pad]) == 0
