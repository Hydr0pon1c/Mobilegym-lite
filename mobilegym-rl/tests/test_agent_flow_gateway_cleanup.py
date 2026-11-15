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

import asyncio
from unittest.mock import Mock

import pytest

from rllm.agents.agent import Episode, Trajectory
from rllm.experimental.engine import agent_flow_engine
from rllm.experimental.engine.agent_flow_engine import AgentFlowEngine
from rllm.experimental.eval.types import EvalOutput


class _Gateway:
    delete_session_after_use = True

    def __init__(self):
        self.deleted = []

    async def acreate_session(self, uid, is_validation=False):
        return uid

    def get_session_url(self, uid):
        return f"http://gateway/{uid}/v1"

    async def aget_traces(self, uid):
        return []

    async def adelete_session(self, uid):
        self.deleted.append(uid)
        return 0


class _DeleteFailingGateway(_Gateway):
    async def adelete_session(self, uid):
        self.deleted.append(uid)
        raise RuntimeError("delete failed")


class _Agent:
    async def arun(self, task, config):
        return Episode(trajectories=[Trajectory()])


class _FailingAgent:
    async def arun(self, task, config):
        raise RuntimeError("rollout failed")


class _Evaluator:
    def evaluate(self, task, episode):
        return EvalOutput(reward=1.0, is_correct=True)


def _make_engine(agent, gateway):
    engine = AgentFlowEngine(
        agent_flow=agent,
        evaluator=_Evaluator(),
        gateway=gateway,
        model="test-model",
        n_parallel_tasks=1,
    )
    # Trace conversion is covered elsewhere; isolate session lifecycle here.
    engine._enrich_episode = lambda episode, traces, uid, task: episode
    return engine


def test_consumed_gateway_session_is_deleted():
    gateway = _Gateway()
    engine = _make_engine(_Agent(), gateway)
    try:
        episode = asyncio.run(engine._run_single({"task": "x"}, "session-1"))
    finally:
        engine.shutdown()

    assert episode.is_correct
    assert gateway.deleted == ["session-1"]


def test_gateway_session_is_deleted_when_rollout_fails():
    gateway = _Gateway()
    engine = _make_engine(_FailingAgent(), gateway)
    try:
        with pytest.raises(RuntimeError, match="rollout failed"):
            asyncio.run(engine._run_single({"task": "x"}, "session-2"))
    finally:
        engine.shutdown()

    assert gateway.deleted == ["session-2"]


def test_gateway_cleanup_failure_does_not_discard_rollout(monkeypatch):
    # Accelerator imports may reconfigure global logging during collection.
    # Check that this branch emits its warning independently of those settings.
    warning = Mock()
    monkeypatch.setattr(agent_flow_engine.logger, "warning", warning)
    gateway = _DeleteFailingGateway()
    engine = _make_engine(_Agent(), gateway)
    try:
        episode = asyncio.run(engine._run_single({"task": "x"}, "session-3"))
    finally:
        engine.shutdown()

    assert episode.is_correct
    assert gateway.deleted == ["session-3"]
    warning.assert_called_once()
    assert "Failed to delete consumed gateway session" in warning.call_args.args[0]
