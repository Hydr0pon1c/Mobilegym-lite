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

"""CPU checks of episode balancing in the real GRPO→GSPO loss path."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.config.actor import ActorConfig

ROOT=Path(__file__).resolve().parents[1]
ns=dict(np=np,torch=torch,DataProto=DataProto,core_algos=core_algos,
        AdvantageEstimator=core_algos.AdvantageEstimator)
def extract(path,names,class_name=None):
    nodes=ast.parse((ROOT/path).read_text()).body
    if class_name:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    chosen=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in chosen}==set(names)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*chosen],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
extract('verl/verl/trainer/ppo/ray_trainer.py',['compute_advantage'])
extract('rllm/experimental/verl/verl_advantage.py',['compute_advantage_verl','_stepwise_advantage_broadcast','_balance_episode_advantages'])
extract('rllm/experimental/verl/verl_backend.py',['_pad_dataproto_to_world_size'],'VerlBackend')

def batch(lengths=(1,3),raw_steps=None):
    n=sum(lengths);episodes=[];last=[];rewards=[];steps=[]
    for i,length in enumerate(lengths):
        episodes.extend([str(i)]*length);last.extend([False]*(length-1)+[True]);rewards.extend([1.-i]*length)
        steps.extend([(raw_steps or lengths)[i]]*length)
    mask=torch.tensor([[1.,float(i%2==0),float(i%3==0)] for i in range(n)])
    return DataProto.from_dict(tensors={'prompts':torch.ones(n,2,dtype=torch.long),'responses':torch.ones(n,3,dtype=torch.long),
        'response_mask':mask,'traj_rewards':torch.tensor([[0.,0.,r] for r in rewards]),'step_rewards':torch.zeros(n,3)},
        non_tensors={'trajectory_ids':np.array(['task_agent']*n),'episode_ids':np.array(episodes),'step_nums':np.array(steps),
        'is_last_step':np.array(last),'is_pad_step':np.zeros(n,dtype=bool),'is_valid':np.ones(n,dtype=bool)})

def config(mode='response_mean',normalize=False):
    return OmegaConf.create({'rllm':{'stepwise_advantage':{'mode':'broadcast','episode_weighting':mode,'normalize_by_steps':normalize}},
        'algorithm':{'adv_estimator':'grpo','gamma':1.,'lam':1.,'norm_adv_by_std_in_grpo':True,'use_kl_in_reward':False},
        'actor_rollout_ref':{'rollout':{'n':2}}})

def gradients(data):
    lp=torch.zeros_like(data.batch['advantages'],requires_grad=True)
    loss,_=core_algos.compute_policy_loss_gspo(torch.zeros_like(lp),lp,data.batch['advantages'],data.batch['response_mask'],
        config=ActorConfig(strategy='fsdp',rollout_n=2,use_dynamic_bsz=True,loss_agg_mode='seq-mean-token-mean',clip_ratio_low=.2,clip_ratio_high=.28))
    loss.backward();return lp.grad

@pytest.mark.parametrize('world_size',[1,4,16])
def test_actual_gspo_equal_episode_totals_with_padding_reordering_and_unequal_token_masks(world_size):
    b=batch();fake=SimpleNamespace(_get_dp_world_size=lambda:world_size,_remove_padding=lambda b:b.select_idxs(np.where(~b.non_tensor_batch['is_pad_step'])[0]))
    b=ns['_pad_dataproto_to_world_size'](fake,b);b.reorder(torch.arange(len(b)-1,-1,-1))
    b,m=ns['compute_advantage_verl'](b,config('episode_mean'))
    assert m['advantage/valid_responses']==4 and m['advantage/episodes']==2
    assert m['advantage/mean_responses_per_episode']==2 and m['advantage/response_weight_mean']==pytest.approx(1)
    assert m['advantage/response_weight_min']==pytest.approx(2/3) and m['advantage/response_weight_max']==2
    grad=gradients(b);pad=torch.as_tensor(b.non_tensor_batch['is_pad_step']);assert torch.count_nonzero(grad[pad])==0
    totals=[grad[torch.as_tensor(b.non_tensor_batch['episode_ids']==str(i))&~pad].sum().item() for i in [0,1]]
    assert totals[0]<0<totals[1] and abs(totals[0])==pytest.approx(abs(totals[1]))

def test_legacy_default_keeps_three_to_one_length_exposure_and_absent_field_behavior():
    cfg=config();del cfg.rllm.stepwise_advantage.episode_weighting
    b,m=ns['compute_advantage_verl'](batch(),cfg);assert m=={}
    g=gradients(b);a=g[torch.as_tensor(b.non_tensor_batch['episode_ids']=='0')].sum().abs()
    z=g[torch.as_tensor(b.non_tensor_batch['episode_ids']=='1')].sum().abs();assert z.item()==pytest.approx(3*a.item())

def test_counts_transformed_responses_instead_of_raw_gui_step_metadata():
    b,m=ns['compute_advantage_verl'](batch(raw_steps=(50,100)),config('episode_mean'))
    assert m['advantage/mean_responses_per_episode']==2
    g=gradients(b);a=g[torch.as_tensor(b.non_tensor_batch['episode_ids']=='0')].sum()
    z=g[torch.as_tensor(b.non_tensor_batch['episode_ids']=='1')].sum();assert a.item()==pytest.approx(-z.item())

def test_single_turn_episodes_are_unchanged_and_returns_stay_consistent():
    a,_=ns['compute_advantage_verl'](batch((1,1)),config())
    b,m=ns['compute_advantage_verl'](batch((1,1)),config('episode_mean'))
    assert torch.equal(a.batch['advantages'],b.batch['advantages'])
    assert torch.equal(a.batch['returns'],b.batch['returns'])
    assert m['advantage/response_weight_mean']==1

@pytest.mark.parametrize('mode,normalize',[('unknown',False),('episode_mean',True)])
def test_invalid_or_double_normalization_fails(mode,normalize):
    with pytest.raises(ValueError):ns['compute_advantage_verl'](batch(),config(mode,normalize))
