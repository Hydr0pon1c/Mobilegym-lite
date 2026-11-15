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

"""
Verl-native advantage computation -- serve as opt-in for verl backend to replace
the default rLLM-native advantage computation.
"""

import numpy as np
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage


def compute_advantage_verl(batch: DataProto, config: DictConfig) -> tuple[DataProto, dict]:
    """Verl-native advantage computation."""
    assert config.rllm.stepwise_advantage.mode == "broadcast", "Only broadcast mode is supported in experimental unified trainer."
    weighting = config.rllm.stepwise_advantage.get("episode_weighting", "response_mean")
    if weighting not in {"response_mean", "episode_mean"}:
        raise ValueError(f"Unsupported episode_weighting: {weighting}")
    if weighting == "episode_mean" and config.rllm.stepwise_advantage.get("normalize_by_steps", False):
        raise ValueError("episode_mean already balances episodes; normalize_by_steps must be false")
    metrics = {}
    batch.non_tensor_batch["uid"] = batch.non_tensor_batch["trajectory_ids"]

    batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

    if config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch,
            kl_ctrl=config.kl_ctrl_in_reward,  # type: ignore[arg-type]
            kl_penalty=config.algorithm.kl_penalty,
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    is_last_step = batch.non_tensor_batch["is_last_step"]
    last_step_indices = np.where(is_last_step)[0]
    not_last_step_indices = np.where(~is_last_step)[0]

    adv_kwargs = dict(
        adv_estimator=config.algorithm.adv_estimator,
        gamma=config.algorithm.gamma,
        lam=config.algorithm.lam,
        num_repeat=config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=config.algorithm.norm_adv_by_std_in_grpo,
        config=config.algorithm,
    )

    if len(not_last_step_indices) == 0:
        # Single-turn episodes need no broadcast.
        batch = compute_advantage(batch, **adv_kwargs)
    else:
        # Compute terminal advantages, then broadcast to earlier responses.
        non_last_step_batch = batch.select_idxs(not_last_step_indices)
        batch = compute_advantage(batch.select_idxs(last_step_indices), **adv_kwargs)
        _stepwise_advantage_broadcast(batch, non_last_step_batch, config)
        batch = DataProto.concat([batch, non_last_step_batch])

    if weighting == "episode_mean":
        metrics.update(_balance_episode_advantages(batch))

    return batch, metrics


def _balance_episode_advantages(batch: DataProto) -> dict[str, float]:
    """Weight each valid response by mean episode length / its episode length.

    Count actual response rows, not raw GUI steps. Padding and empty responses
    do not contribute. This preserves mean response weight 1, not gradient norm
    or Adam update size. Sequence losses still use their existing minibatches.
    """
    mask = batch.batch["response_mask"].bool()
    active = mask.any(dim=-1).detach().cpu().numpy()
    if "is_pad_step" in batch.non_tensor_batch:
        active &= ~batch.non_tensor_batch["is_pad_step"].astype(bool)
    keys = list(zip(batch.non_tensor_batch["trajectory_ids"], batch.non_tensor_batch["episode_ids"], strict=True))
    counts = {}
    for key, keep in zip(keys, active, strict=True):
        if keep:
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError("episode_mean requires at least one valid response")
    n_rows = int(active.sum())
    mean_length = n_rows / len(counts)
    weights = batch.batch["advantages"].new_tensor(
        [mean_length / counts[key] if keep else 0.0 for key, keep in zip(keys, active, strict=True)]
    )
    batch.batch["advantages"] = batch.batch["advantages"] * weights[:, None]
    if "returns" in batch.batch:
        batch.batch["returns"] = batch.batch["returns"] * weights[:, None]
    live_weights = weights[torch.as_tensor(active, device=weights.device)]
    return {
        "advantage/episode_mean_enabled": 1.0,
        "advantage/valid_responses": float(n_rows),
        "advantage/episodes": float(len(counts)),
        "advantage/mean_responses_per_episode": float(mean_length),
        "advantage/response_weight_mean": float(live_weights.mean().item()),
        "advantage/response_weight_min": float(live_weights.min().item()),
        "advantage/response_weight_max": float(live_weights.max().item()),
    }


def _stepwise_advantage_broadcast(last_step_batch: DataProto, non_last_step_batch: DataProto, config: DictConfig) -> None:
    """Broadcast advantage from last step to all other steps."""
    src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
    src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
    src_steps = last_step_batch.non_tensor_batch["step_nums"]
    src_mask = last_step_batch.batch["response_mask"]
    src_advantages = last_step_batch.batch["advantages"]

    tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
    tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
    tgt_mask = non_last_step_batch.batch["response_mask"]

    traj_ep_to_scalar_adv = {}
    for i, (traj_id, eps_id) in enumerate(zip(src_traj_ids, src_eps_ids, strict=False)):
        mask = src_mask[i].bool()
        scalar = src_advantages[i][mask].mean()

        if config.rllm.stepwise_advantage.get("normalize_by_steps", False):
            scalar = scalar / src_steps[i]
            last_step_batch.batch["advantages"][i][mask] = scalar

        traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

    scalar_rows = torch.stack(
        [torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))]
    )

    final_advantage = scalar_rows * tgt_mask
    non_last_step_batch.batch["advantages"] = final_advantage
    non_last_step_batch.batch["returns"] = final_advantage
