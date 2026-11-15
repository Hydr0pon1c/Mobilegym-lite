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

"""Exercise startup method bodies without importing the GPU dependency stack."""
import ast
import asyncio
import errno
import json
import logging
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def source_method(relative_path, class_name, method_name):
    # Importing AgentLoopManager eagerly imports Torch/Transformers and performs
    # accelerator discovery. Execute the exact source methods with only their
    # CPU dependencies instead; these tests do not validate Ray/GPU integration.
    path = ROOT / "mobilegym-rl/verl/verl" / relative_path
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"asyncio": asyncio, "logger": logging.getLogger(__name__)}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


initialize_servers = source_method("experimental/agent_loop/agent_loop.py", "AgentLoopManager", "_initialize_llm_servers")
validate_rollout = source_method("workers/config/rollout.py", "RolloutConfig", "__post_init__")


def manager(limit, initialize, *, mode="hybrid", replicas=16):
    class Replica:
        def __init__(self, replica_rank, **kwargs):
            self.rank = replica_rank
            self._server_handle = f"handle-{replica_rank}"
            self._server_address = f"address-{replica_rank}"

        async def init_hybrid(self, workers):
            assert mode == "hybrid"
            await initialize(self)

        async def init_hybrid_colocated(self, workers, pool):
            assert mode == "colocated"
            await initialize(self)

        async def init_standalone(self):
            assert mode == "standalone"
            await initialize(self)

    return SimpleNamespace(
        rollout_config=OmegaConf.create({
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "trtllm" if mode == "colocated" else "vllm",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "n_gpus_per_node": replicas,
            "nnodes": 1,
            "server_init_concurrency": limit,
            "prometheus": {"enable": False},
        }),
        worker_group=None if mode == "standalone" else SimpleNamespace(world_size=replicas),
        rollout_resource_pool=object(),
        rollout_replica_class=Replica,
        model_config=object(),
    )


@pytest.mark.parametrize("mode", ["hybrid", "colocated", "standalone"])
@pytest.mark.parametrize("limit", [0, 1, 2])
def test_startup_limit_preserves_all_serving_replicas(limit, mode):
    active = peak = 0

    async def initialize(replica):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1

    instance = manager(limit, initialize, mode=mode)
    asyncio.run(initialize_servers(instance))
    assert peak == (limit or 16)
    assert instance.server_handles == [f"handle-{i}" for i in range(16)]
    assert instance.server_addresses == [f"address-{i}" for i in range(16)]


@pytest.mark.parametrize("limit", [0, 1])
def test_delayed_socket_bind_race(limit):
    # Reproduce probe -> close -> child startup -> bind, as in vLLM's
    # multiprocess executor. Each replica probes the same candidate sequence.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        first_port = probe.getsockname()[1]
    listeners = []

    async def initialize(replica):
        port = first_port
        while True:
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                    break
                except OSError as exc:
                    if exc.errno != errno.EADDRINUSE:
                        raise
                    port += 1
        await asyncio.sleep(0.001)
        listener = socket.socket()
        listeners.append(listener)
        listener.bind(("127.0.0.1", port))
        listener.listen()

    instance = manager(limit, initialize)
    try:
        if limit == 0:
            with pytest.raises(OSError) as caught:
                asyncio.run(initialize_servers(instance))
            assert caught.value.errno == errno.EADDRINUSE
        else:
            asyncio.run(initialize_servers(instance))
            assert len({listener.getsockname()[1] for listener in listeners}) == 16
            assert len(instance.server_handles) == 16
    finally:
        for listener in listeners:
            listener.close()


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_startup_limit(value):
    with pytest.raises(ValueError, match="server_init_concurrency"):
        validate_rollout(SimpleNamespace(server_init_concurrency=value))


def test_initialization_error_still_fails_training():
    async def initialize(replica):
        raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError, match="engine failed"):
        asyncio.run(initialize_servers(manager(1, initialize)))


def test_public_training_enables_serial_startup():
    spec = json.loads((ROOT / "configs/training/feature0.json").read_text())
    assert spec["overrides"]["actor_rollout_ref.rollout.server_init_concurrency"] == 1
    assert spec["overrides"]["train_completeness"] == "feature0"
    assert spec["overrides"]["reference_bundle"] is None
    defaults = OmegaConf.load(ROOT / "mobilegym-rl/verl/verl/trainer/config/rollout/rollout.yaml")
    assert defaults.server_init_concurrency == 0
