# Copyright 2026 OPPO
#
# Modified by OPPO for this release.
# OPPO copyright applies only to the modifications.
#

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
from typing import Optional

import torch


def normalize_uniform_temperature(temperature: float | torch.Tensor) -> float:
    """Return a scalar temperature when every sample uses the same value.

    The memory-efficient LM-head computes a packed batch in chunks and therefore
    currently supports one sampling temperature per forward.  RLLM stores that
    common value as a per-sample tensor, so accept the tensor representation while
    rejecting genuinely mixed-temperature batches.
    """
    if isinstance(temperature, torch.Tensor):
        values = temperature.detach().reshape(-1)
        if values.numel() == 0:
            raise ValueError("temperature tensor must not be empty")
        first = values[0]
        if not torch.all(values == first).item():
            raise ValueError("fused LM-head requires a uniform temperature within each micro-batch")
        temperature = first.item()

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    return temperature


def _fused_linear_for_ppo_fwd(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    logits = (hidden_states @ vocab_weights.t()) / temperature
    logits = logits.to(torch.float32)

    # Slower but more numerically stable to do log_softmax than probs.log()
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)

    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)

    # Keep probability-domain outputs in FP32.  Casting them back to BF16 can
    # quantize low-probability tokens by 0.1 or more, which is large enough to
    # trigger GSPO clipping and create an artificial rollout/actor mismatch.
    return token_log_probs, entropy


def _fused_linear_for_ppo_bwd(
    dlog_probs: Optional[torch.FloatTensor],
    dentropy: Optional[torch.FloatTensor],
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)

    dlogits = 0

    # Gradient from log_probs
    if dlog_probs is not None:
        one_hot_input = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
        dlogits += dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)

    # Gradient from entropy
    if dentropy is not None:
        log_probs = logits.log_softmax(dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        dlogits += probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))

    dlogits = dlogits.to(orig_dtype) / temperature

    dhidden_states = dlogits @ vocab_weights
    dvocab_weights = dlogits.t() @ hidden_states

    return dhidden_states, dvocab_weights


class FusedLinearForPPOFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        ctx.set_materialize_grads(False)

        # Cast to a 2D tensor of the shape [T, D] for ease of working
        orig_ndim = hidden_states.ndim
        assert orig_ndim in (2, 3), f"Invalid hidden_states shape, received {hidden_states.shape}"

        orig_batch_size = -1
        if orig_ndim == 3:
            assert input_ids.ndim == 2, f"input_ids shape doesn't match, {hidden_states.shape} {input_ids.shape}"
            orig_batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.flatten(0, 1)
            input_ids = input_ids.flatten(0, 1)

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad
        log_probs = torch.zeros(
            T,
            dtype=torch.float32,
            device=hidden_states.device,
            requires_grad=output_requires_grad,
        )
        entropy = torch.zeros(
            T,
            dtype=torch.float32,
            device=hidden_states.device,
            requires_grad=output_requires_grad,
        )

        # Perform forward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)

            chunk_log_probs, chunk_entropy = _fused_linear_for_ppo_fwd(
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
            )
            log_probs[chunk_start:chunk_end] = chunk_log_probs
            entropy[chunk_start:chunk_end] = chunk_entropy

        # Cast the output back to the original input dimension
        if orig_ndim == 3:
            log_probs = log_probs.view(orig_batch_size, -1)
            entropy = entropy.view(orig_batch_size, -1)

        ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
        ctx.orig_batch_size = orig_batch_size
        ctx.orig_ndim = orig_ndim
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size

        return log_probs, entropy

    @staticmethod
    def backward(ctx, dlog_probs: Optional[torch.FloatTensor], dentropy: Optional[torch.FloatTensor]):
        assert dlog_probs is not None or dentropy is not None

        hidden_states, vocab_weights, input_ids = ctx.saved_tensors
        orig_batch_size = ctx.orig_batch_size
        orig_ndim = ctx.orig_ndim
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size

        # Here orig_ndim refers to the orig_ndim of hidden_states
        if orig_ndim == 3:
            if dlog_probs is not None:
                dlog_probs = dlog_probs.flatten()
            if dentropy is not None:
                dentropy = dentropy.flatten()

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        dhidden_states = None
        if hidden_states.requires_grad:
            dhidden_states = torch.zeros_like(hidden_states)
        dvocab_weights = None
        if vocab_weights.requires_grad:
            dvocab_weights = torch.zeros_like(vocab_weights)

        # Perform backward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_dlog_probs = None
            if dlog_probs is not None:
                chunk_dlog_probs = dlog_probs[chunk_start:chunk_end]
            chunk_dentropy = None
            if dentropy is not None:
                chunk_dentropy = dentropy[chunk_start:chunk_end]

            h, v = _fused_linear_for_ppo_bwd(
                dlog_probs=chunk_dlog_probs,
                dentropy=chunk_dentropy,
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
            )

            if hidden_states.requires_grad:
                dhidden_states[chunk_start:chunk_end] += h
            if vocab_weights.requires_grad:
                dvocab_weights += v

        # Cast the output back to the original input dimension
        if orig_ndim == 3 and hidden_states.requires_grad:
            hidden_size = hidden_states.shape[-1]
            dhidden_states = dhidden_states.view(orig_batch_size, -1, hidden_size)

        return (
            dhidden_states,  # hidden_states
            dvocab_weights,  # vocab_weights
            None,  # input_ids
            None,  # temperature
            None,  # chunk_size
        )


class FusedLinearForPPO(torch.nn.Module):
    def __init__(self, chunk_size: int = 512):
        super().__init__()

        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        input_ids = input_ids.to(torch.int64)
        temperature = normalize_uniform_temperature(temperature)
        return FusedLinearForPPOFunction.apply(
            hidden_states,
            vocab_weights,
            input_ids,
            temperature,
            self.chunk_size,
        )
