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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rllm.experimental.rollout.verl_engine import _gateway_model_name
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _register_tensor_lora_with_openai,
)


def _config(*, rollout_name="vllm", rank=0, nested_rank=0, merge=False):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"name": rollout_name},
                "model": {
                    "lora_rank": rank,
                    "lora": {"rank": nested_rank, "merge": merge},
                },
            }
        }
    )


@pytest.mark.parametrize("rank,nested_rank", [(16, 0), (0, 16)])
def test_vllm_unmerged_lora_uses_adapter_model(rank, nested_rank):
    assert _gateway_model_name(_config(rank=rank, nested_rank=nested_rank)) == VLLM_LORA_NAME


@pytest.mark.parametrize(
    "config",
    [
        _config(rank=0),
        _config(rank=16, merge=True),
        _config(rollout_name="sglang", rank=16),
    ],
)
def test_non_adapter_rollout_keeps_configured_model(config):
    assert _gateway_model_name(config) is None


def test_tensor_lora_is_registered_with_openai_frontend():
    serving_models = SimpleNamespace(lora_requests={})
    state = SimpleNamespace(openai_serving_models=serving_models)

    request = _register_tensor_lora_with_openai(state, "/models/base")

    assert serving_models.lora_requests == {VLLM_LORA_NAME: request}
    assert request.lora_name == VLLM_LORA_NAME
    assert request.lora_int_id == VLLM_LORA_INT_ID
    assert request.lora_path == VLLM_LORA_PATH
    assert request.base_model_name == "/models/base"


def test_missing_openai_registry_fails_loudly():
    with pytest.raises(RuntimeError, match="registry is unavailable"):
        _register_tensor_lora_with_openai(SimpleNamespace(), "/models/base")
