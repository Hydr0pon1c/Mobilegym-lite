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

"""Regression for the shared Qwen-VL cleanup removed in b6d7c2e.

Exercise real rLLM batching, M-RoPE, DataProto -> TensorDict no-padding and
VERL collation. No Ray cluster, model weights, browser or accelerator required.
"""
import copy

import pytest
import torch

from rllm.experimental.verl.dataclass import AccumulatedData, ProcessedStepData
from rllm.experimental.verl.transform import _batch_tensors_and_build_data_proto
from rllm.workflows.workflow import TerminationReason
from verl.utils.model import extract_multi_modal_inputs
from verl.workers.utils.padding import left_right_2_no_padding


def processor_stub():
    # Dispatch on the same class names as real Transformers 4/5 processors.
    image = type("Qwen2VLImageProcessor", (), {"merge_size": 2})()
    return type("Qwen3VLProcessor", (), {
        "image_processor": image, "image_token_id": 20,
        "video_token_id": 21, "vision_start_token_id": 19,
    })()


def accumulated_inputs(with_types=True, text_only=False):
    accumulated = AccumulatedData()
    for i, tail in enumerate(([12], [12, 13, 14, 15])):
        prompt = torch.tensor([10, 19, 20, 20, 20, 20, 22] + tail)
        visual = {"pixel_values": torch.full((16, 6), float(i)),
                  "image_grid_thw": torch.tensor([[1, 4, 4]])}
        if with_types:
            visual["mm_token_type_ids"] = (prompt == 20).long().unsqueeze(0)
        if text_only:
            prompt, visual = torch.tensor([10] + tail), {}
        step = ProcessedStepData(
            prompt=prompt, response=torch.tensor([16, 17]), mask=torch.ones(2),
            step_reward=float(i), step_id=str(i), multi_modal_inputs=visual,
            logprobs=[-0.2, -0.3],
        )
        accumulated.add_step(step, str(i), float(i), 1, True)
        accumulated.episode_ids.append(str(i))
        accumulated.is_correct.append(bool(i))
        accumulated.termination_reasons.append(TerminationReason.ENV_DONE)
        accumulated.metrics.append({})
        accumulated.repeat_counts.append(1)
    return accumulated


def batch(accumulated, processor=None):
    return _batch_tensors_and_build_data_proto(accumulated, 0, 16, 4, processor)


@pytest.mark.parametrize("with_types", [True, False])
def test_qwen3vl_variable_prompts_preserve_inputs_and_positions(with_types):
    accumulated = accumulated_inputs(with_types)
    expected = batch(accumulated_inputs(False), processor_stub())
    before = copy.deepcopy(accumulated.multi_modal_inputs)
    actual = batch(accumulated, processor_stub())
    assert actual.batch["position_ids"].shape == (2, 4, 20)
    for key in expected.batch.keys():
        torch.testing.assert_close(actual.batch[key], expected.batch[key])
    for i, visual in enumerate(actual.non_tensor_batch["multi_modal_inputs"]):
        assert "mm_token_type_ids" not in visual
        for key in ("pixel_values", "image_grid_thw"):
            torch.testing.assert_close(visual[key], before[i][key])
    # Same conversion and collation as actor compute_log_prob / update_actor.
    packed = left_right_2_no_padding(actual.to_tensordict())
    visual = extract_multi_modal_inputs(packed["multi_modal_inputs"])
    assert visual["pixel_values"].shape == (32, 6)
    assert visual["image_grid_thw"].shape == (2, 3)
    assert "mm_token_type_ids" not in visual
    torch.testing.assert_close(visual["images_seqlens"], torch.tensor([16, 16]))


def test_cleanup_happens_after_position_ids(monkeypatch):
    import rllm.experimental.verl.transform as transform
    original = transform._handle_multimodal_position_ids

    def positions(processor, input_ids, attention_mask, multi_modal_inputs):
        assert all("mm_token_type_ids" in v for v in multi_modal_inputs)
        return original(processor, input_ids, attention_mask, multi_modal_inputs)

    monkeypatch.setattr(transform, "_handle_multimodal_position_ids", positions)
    batch(accumulated_inputs(), processor_stub())


def test_text_only_batch_unchanged():
    result = batch(accumulated_inputs(text_only=True))
    assert "multi_modal_inputs" not in result.non_tensor_batch
    assert result.batch["position_ids"].shape == (2, 20)


def test_bad_pixel_shapes_still_fail_instead_of_being_dropped():
    accumulated = accumulated_inputs()
    accumulated.multi_modal_inputs[1]["pixel_values"] = torch.zeros(16, 7)
    result = batch(accumulated, processor_stub())
    with pytest.raises(RuntimeError, match="Sizes of tensors"):
        extract_multi_modal_inputs(result.non_tensor_batch["multi_modal_inputs"])
