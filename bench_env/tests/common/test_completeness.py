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

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bench_env.completeness import APPS, TASK_IDS, apply_resources, project_resources
from bench_env.completeness_check import matching_controls
from bench_env.config import RunnerConfig
from bench_env.runner.base import Controller
from bench_env.rerun import _collect_cli_overrides
from bench_env.task.registry import TaskRegistry


def task(tid, **params):
    return SimpleNamespace(id=tid, apps=[tid.split(".")[0]], params=params, answer_fields=None)


def wechat_state():
    return {"apps": {"wechat": {
        "user": {"wxid": "self"},
        "contacts": [{"wxid": str(i), "name": f"friend{i}", "region": f"region{i}"} for i in range(6)],
        "chats": [{"id": "0", "messages": [{"text": "old"}]}, {"id": "self", "messages": []}],
        "moments": [{"id": "m0", "wxid": "0"}, {"id": "me", "wxid": "self"}],
    }}}


def test_scope_is_the_explicit_six_app_v3_snapshot():
    ids = Path(__file__).resolve().parents[2] / "splits/env_effect_v3_train.txt"
    registry = TaskRegistry()
    expected = {tid for tid in ids.read_text().splitlines()
                if tid and not tid.startswith("#") and set(registry.create_task(tid).apps) & APPS}
    assert TASK_IDS == expected
    assert len(TASK_IDS) == 95


def test_nested_masks_preserve_rows_order_rng_and_original_state():
    original = wechat_state()
    before = deepcopy(original)
    rng = random.getstate()
    t = task("wechat.ReadContactRegion", contact="friend3")
    projections = [project_resources(t, original, m)[0]["apps"]["wechat"]
                   for m in ["resources_0", "resources_1", "resources_2", "resources_full"]]
    kept = [{r["wxid"] for r in s["contacts"]} for s in projections]
    assert kept[0] == {"3"} and [len(ids) for ids in kept] == [1, 2, 3, 6]
    assert kept[0] < kept[1] < kept[2] < kept[3]
    assert projections[0]["chats"] == [{"id": "self", "messages": []}]  # no new shortcut chat
    assert projections[0]["moments"] == [{"id": "me", "wxid": "self"}]
    for result in projections:
        ids = {c["wxid"] for c in result["contacts"]}
        assert result["contacts"] == [c for c in original["apps"]["wechat"]["contacts"] if c["wxid"] in ids]
    assert original == before and random.getstate() == rng
    t.id += "_i7"
    assert project_resources(t, original, "resources_1")[0]["apps"]["wechat"] == projections[1]
    projections[1]["contacts"][0]["name"] = "mutated"
    assert original == before


def test_missing_target_and_duplicate_ids_fail_instead_of_removing_the_goal():
    t = task("wechat.ReadContactRegion", contact="missing")
    with pytest.raises(ValueError, match="did not resolve"):
        project_resources(t, wechat_state(), "resources_0")
    s = wechat_state()
    s["apps"]["wechat"]["contacts"].append(deepcopy(s["apps"]["wechat"]["contacts"][0]))
    with pytest.raises(ValueError, match="Duplicate IDs"):
        project_resources(task("wechat.ReadContactRegion", contact="friend0"), s, "resources_0")


def test_latest_prefix_keeps_income_rows_and_counting_keeps_nonmatches():
    rows = [{"id": str(i), "timestamp": i, "delta": i if i % 2 else -i, "counterpartyName": "转账-A"}
            for i in range(9)]
    state = {"apps": {"alipay": {"transferRecords": rows}}}
    patch, _ = project_resources(task("alipay.AnalyzeSpending"), state, "resources_0")
    assert patch["apps"]["alipay"]["transferRecords"] == rows[-5:]
    patch, record = project_resources(task("alipay.CountLargeTransferIncomes", amount=3), state, "resources_0")
    assert patch["apps"]["alipay"]["transferRecords"] == rows
    assert record["effective"] is False


def test_most_participants_retains_all_hosted_candidates_not_only_the_winner():
    rows = [{"id": str(i), "hostId": "me" if i < 3 else "other", "participants": list(range(i))}
            for i in range(5)]
    state = {"apps": {"tencent_meeting": {"user": {"id": "me"}, "history": rows}}}
    patch, _ = project_resources(task("tencent_meeting.FindMeetingWithMostParticipants"), state, "resources_0")
    assert patch["apps"]["tencent_meeting"]["history"] == rows[:3]


def test_unknown_template_is_explicitly_unsupported():
    patch, record = project_resources(task("wechat.FutureTask"), {}, "resources_0")
    assert patch == {"apps": {}} and record["supported"] is False and record["effective"] is False


def test_replay_distinguishes_popup_buttons_and_resolves_input_fields():
    target = {"kind": "label", "label": "删除", "tag": "BUTTON", "context": "取消 删除"}
    popup = {"identity": target}
    page = {"identity": {**target, "context": "日程详情 删除"}}
    assert matching_controls(target, [page, popup]) == [popup]
    field = {"kind": "field", "name": "Search", "tag": "INPUT", "type": "text"}
    assert matching_controls(field, [{"identity": field}, {"identity": {**field, "type": "password"}}]) == [{"identity": field}]
    save = {"kind": "context", "context": "Edit alarm", "tag": "BUTTON", "icon": [["path", "check"]]}
    cancel = {**save, "icon": [["path", "cross"]]}
    assert matching_controls(save, [{"identity": cancel}, {"identity": save}]) == [{"identity": save}]


@pytest.mark.asyncio
async def test_silently_ignored_injection_is_a_setup_error():
    state = wechat_state()
    obs = SimpleNamespace(state=state)
    env = SimpleNamespace(set_state=AsyncMock(), get_observation=AsyncMock(return_value=obs))
    t = task("wechat.ReadContactRegion", contact="friend3")
    with pytest.raises(ValueError, match="readback mismatch"):
        await apply_resources(env, t, obs, "resources_0")
    assert await apply_resources(env, t, obs, None) is obs
    assert t._completeness_record is None


@pytest.mark.asyncio
async def test_controller_projects_after_sampling_and_reset_restores_full_state(monkeypatch):
    monkeypatch.setattr("bench_env.runner.base.prepare_visual_episode", AsyncMock())
    state = wechat_state()
    async def set_state(patch, **kwargs):
        for app, fields in patch["apps"].items():
            state["apps"][app].update(deepcopy(fields))
    async def observation():
        return SimpleNamespace(state=deepcopy(state))
    env = SimpleNamespace(set_state=set_state, get_observation=observation)
    t = task("wechat.ReadContactRegion", contact="not-sampled-yet")
    async def setup(env):
        state.clear()
        state.update(wechat_state())  # normal task reset
        t.params["contact"] = "friend3"  # final _post_sample value
        return await observation()
    t.setup = setup
    reduced, _ = await Controller.setup(env, t, eval_mode="text", completeness="resources_0")
    assert [c["name"] for c in reduced.state["apps"]["wechat"]["contacts"]] == ["friend3"]
    restored, _ = await Controller.setup(env, t, eval_mode="text")
    assert len(restored.state["apps"]["wechat"]["contacts"]) == 6
    assert t._completeness_record is None


def test_cli_config_roundtrip_and_unsupported_modes():
    from bench_env.run import create_parser
    args = create_parser().parse_args(["--agent", "generic", "--model-name", "test", "--completeness", "resources_1"])
    cfg = RunnerConfig.from_args(args)
    assert cfg.to_dict()["completeness"] == "resources_1"
    with pytest.raises(ValueError, match="simulator"):
        RunnerConfig(agent="generic", model_name="test", device="real", completeness="resources_0")
    with pytest.raises(ValueError, match="Unknown completeness"):
        RunnerConfig(agent="generic", model_name="test", completeness="resources_typo")
    with pytest.raises(ValueError, match="in-place rerun"):
        _collect_cli_overrides(argparse.Namespace(completeness="resources_2"))
