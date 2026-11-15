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

"""Lite composes audited projections without changing task semantics or full eval."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from bench_env.completeness import apply_completeness, project_resources
from bench_env.feature_completeness import VERSION as FEATURE_VERSION
from bench_env.runner.base import Controller
from bench_env.tests.common.test_completeness import task, wechat_state


class MemoryEnv:
    def __init__(self, state):
        self.state = deepcopy(state)
        self.disabled = ["clock.timer"]  # Simulate reuse of a previously gated page.
        self.page = SimpleNamespace(evaluate=self.evaluate)

    async def evaluate(self, script, record=None):
        if record is not None:
            self.disabled = list(record["disabled"])
            return {"version": FEATURE_VERSION, "disabled": self.disabled}
        changed = bool(self.disabled)
        self.disabled = []
        return changed

    async def set_state(self, patch, **kwargs):
        for app, fields in patch["apps"].items():
            self.state["apps"][app].update(deepcopy(fields))

    async def get_observation(self):
        return SimpleNamespace(state=deepcopy(self.state))


@pytest.mark.asyncio
async def test_lite_keeps_candidates_answer_and_both_axis_records():
    env = MemoryEnv(wechat_state())
    t = task("wechat.ReadContactRegion", contact="friend3")
    original = await env.get_observation()
    params = deepcopy(t.params)
    expected, _ = project_resources(t, original.state, "resources_2")

    reduced = await apply_completeness(env, t, original, "lite_v1")

    for key, rows in expected["apps"]["wechat"].items():
        assert reduced.state["apps"]["wechat"][key] == rows
    assert reduced.state["apps"]["wechat"]["user"] == original.state["apps"]["wechat"]["user"]
    contacts = reduced.state["apps"]["wechat"]["contacts"]
    assert len(contacts) == 3
    assert next(c["region"] for c in contacts if c["name"] == "friend3") == "region3"
    assert env.disabled == ["wechat.moments", "wechat.profile_edit"]
    assert t.params == params
    record = t._completeness_record
    assert record["mode"] == record["version"] == "lite_v1"
    assert record["resources"]["query_answer_checked"]
    assert record["resources"]["mode"] == "resources_2"
    assert record["features"]["mode"] == "feature0"
    assert record["effective"] and not record["background_only"]
    assert original.state == wechat_state()


@pytest.mark.asyncio
@pytest.mark.parametrize("tid,state,resource_effective,feature_effective", [
    ("alipay.EnableDarkMode", {"apps": {"alipay": {"transferRecords": [{"id": i} for i in range(6)]}}}, True, False),
    ("calendar.CreateEvent", {"apps": {"calendar": {"events": []}}}, False, True),
    ("wechat.FutureTask", wechat_state(), False, False),
])
async def test_lite_supports_single_axis_and_unknown_full_fallback(tid, state, resource_effective, feature_effective):
    env, t = MemoryEnv(state), task(tid)
    reduced = await apply_completeness(env, t, await env.get_observation(), "lite_v1")
    record = t._completeness_record
    assert record["resources"]["effective"] is resource_effective
    assert record["features"]["effective"] is feature_effective
    assert record["effective"] is (resource_effective or feature_effective)
    if not record["supported"]:
        assert reduced.state == state and env.disabled == []


@pytest.mark.asyncio
async def test_controller_lite_then_full_restores_rows_and_masks(monkeypatch):
    from unittest.mock import AsyncMock
    monkeypatch.setattr("bench_env.runner.base.prepare_visual_episode", AsyncMock())
    env, t = MemoryEnv(wechat_state()), task("wechat.ReadContactRegion", contact="friend3")

    async def setup(env):
        env.state = wechat_state()
        return await env.get_observation()

    t.setup = setup
    initial, _ = await Controller.setup(env, t, eval_mode="text", completeness="lite_v1")
    assert len(initial.state["apps"]["wechat"]["contacts"]) == 3 and env.disabled
    full, _ = await Controller.setup(env, t, eval_mode="text")
    assert full.state == wechat_state() and not env.disabled
    assert t._completeness_record is None


@pytest.mark.asyncio
async def test_lite_rejects_feature_side_effects():
    env, t = MemoryEnv(wechat_state()), task("wechat.ReadContactRegion", contact="friend3")
    evaluate = env.evaluate

    async def corrupt(script, record=None):
        actual = await evaluate(script, record)
        if record is not None:
            env.state["apps"]["wechat"]["user"]["wxid"] = "unexpected"
        return actual

    env.page.evaluate = corrupt
    with pytest.raises(ValueError, match="changed app data"):
        await apply_completeness(env, t, await env.get_observation(), "lite_v1")


def test_lite_cli_roundtrip_and_real_device_rejection():
    from bench_env.config import RunnerConfig
    from bench_env.run import create_parser
    args = create_parser().parse_args(["--agent", "generic", "--model-name", "test", "--completeness", "lite_v1"])
    assert RunnerConfig.from_args(args).to_dict()["completeness"] == "lite_v1"
    with pytest.raises(ValueError, match="simulator"):
        RunnerConfig(agent="generic", model_name="test", device="real", completeness="lite_v1")
