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

"""Convert gateway TraceRecord to training-compatible Step, plus shared metrics."""

from copy import deepcopy
import json
from typing import Any

import torch

from rllm_model_gateway.models import TraceRecord

from rllm.agents.agent import Step, Trajectory
from rllm.experimental.rollout import ModelOutput
from rllm.tools.tool_base import ToolCall


def _parse_openai_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert OpenAI-format tool_calls to rLLM ToolCall objects."""
    result = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except (json.JSONDecodeError, ValueError):
                arguments = {"raw": args_raw}
        else:
            arguments = args_raw
        result.append(ToolCall(name=name, arguments=arguments))
    return result


def _contains_image(messages: list[dict[str, Any]]) -> bool:
    """Return whether OpenAI-format chat messages contain an image."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"image", "image_url"} or "image" in item or "image_url" in item:
                return True
    return False


def _normalize_vision_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI nested ``image_url.url`` objects for qwen-vl-utils."""
    normalized = deepcopy(messages)
    for message in normalized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url"):
                item["image_url"] = image_url["url"]
    return normalized


def build_multimodal_inputs(
    messages: list[dict[str, Any]],
    processor,
    *,
    disable_thinking: bool = False,
) -> dict[str, torch.Tensor] | None:
    """Rebuild the visual tensors omitted from gateway traces.

    The gateway records the exact prompt token IDs returned by vLLM, so those
    remain authoritative.  This function only reconstructs the pixel/grid
    tensors required when the actor recomputes log probabilities and gradients.
    """
    if processor is None or not _contains_image(messages):
        return None

    from qwen_vl_utils import process_vision_info

    normalized = _normalize_vision_messages(messages)
    # Some processor versions normalize content dictionaries in place, so
    # extract the image objects before rendering the chat template.
    images, videos = process_vision_info(
        normalized,
        image_patch_size=processor.image_processor.patch_size,
    )
    template_kwargs: dict[str, Any] = {}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    prompt = processor.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    model_inputs = processor(
        text=[prompt],
        images=images,
        videos=videos,
        return_tensors="pt",
    )
    model_inputs.pop("input_ids", None)
    model_inputs.pop("attention_mask", None)
    multi_modal_inputs = dict(model_inputs)

    # The actor casts pixels to the vision tower dtype.  Storing them as bf16
    # first halves host memory and transfer volume without changing that input.
    for key in ("pixel_values", "pixel_values_videos"):
        value = multi_modal_inputs.get(key)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            multi_modal_inputs[key] = value.to(torch.bfloat16)

    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        multi_modal_inputs["images_seqlens"] = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        )

    if "pixel_values" not in multi_modal_inputs or "image_grid_thw" not in multi_modal_inputs:
        raise ValueError("Image trace did not produce pixel_values and image_grid_thw")
    return multi_modal_inputs


def trace_record_to_step(
    trace: TraceRecord,
    *,
    processor=None,
    disable_thinking: bool = False,
) -> Step:
    """Convert a gateway TraceRecord to a training Step.

    TraceRecord has clean top-level fields from vLLM:
    - prompt_token_ids
    - completion_token_ids
    - logprobs (per-token)
    """
    content = trace.response_message.get("content", "") or ""
    reasoning = trace.response_message.get("reasoning", "") or ""

    # Extract tool_calls from response message (OpenAI format)
    raw_tool_calls = trace.response_message.get("tool_calls")
    tool_calls = _parse_openai_tool_calls(raw_tool_calls) if raw_tool_calls else None

    multi_modal_inputs = build_multimodal_inputs(
        trace.messages,
        processor,
        disable_thinking=disable_thinking,
    )

    model_output = ModelOutput(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        prompt_ids=trace.prompt_token_ids,
        completion_ids=trace.completion_token_ids,
        multi_modal_inputs=multi_modal_inputs,
        logprobs=trace.logprobs or [],
        prompt_length=len(trace.prompt_token_ids),
        completion_length=len(trace.completion_token_ids),
        finish_reason=trace.finish_reason,
    )

    # Build chat_completions: input messages + assistant response
    chat_completions = list(trace.messages)
    chat_completions.append(trace.response_message)

    return Step(
        id=trace.trace_id,
        chat_completions=chat_completions,
        model_output=model_output,
        model_response=content,
        thought=reasoning,
        metadata=trace.metadata,
    )


def compute_step_metrics(trajectories: list[Trajectory]) -> dict:
    """Standard training metrics from trajectories (shared by local and remote engines)."""
    all_response_lens = [len(s.response_ids) for t in trajectories for s in t.steps]
    all_prompt_lens = [len(s.prompt_ids) for t in trajectories for s in t.steps]
    return {
        "num_trajectories": len(trajectories),
        "steps_used": sum(len(t.steps) for t in trajectories),
        "mean_response_len": (sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0),
        "max_response_len": max(all_response_lens, default=0),
        "min_response_len": min(all_response_lens, default=0),
        "max_prompt_len": max(all_prompt_lens, default=0),
        "min_prompt_len": min(all_prompt_lens, default=0),
    }
