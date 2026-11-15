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

"""Native GUI behavior in Lite/layout modes; run with -m live on a capable frontend."""
from types import SimpleNamespace

import pytest

from bench_env.completeness_check import make_task
from bench_env.runner.base import Controller, Evaluator

pytestmark = [pytest.mark.live, pytest.mark.asyncio(loop_scope="session")]


@pytest.mark.parametrize("path", ["edit", "long_press", "cancel_wrong_then_edit", "delete_wrong"])
@pytest.mark.parametrize("mode", ["lite_v1", "feature0_layout_v1"])
async def test_alarm_alternatives_recovery_and_wrong_object_judging(env, path, mode):
    task = make_task("clock.DeleteAlarm", 41)
    initial, _ = await Controller.setup(env, task, eval_mode="text", completeness=mode)
    alarms = initial.state["apps"]["clock"]["alarms"]
    assert len(alarms) == (3 if mode == 'lite_v1' else 7)
    target = next(a for a in alarms if a["id"] == task.params["alarm_id"])
    other = next(a for a in alarms if a["id"] != target["id"])
    page = env.page

    async def row(alarm):
        label = f'{alarm["hour"]:02d}:{alarm["minute"]:02d}'
        element = page.locator('[data-action="alarm.item.selection.enter"]').filter(
            has=page.get_by_text(label, exact=True))
        # Gesture binders can expose either trigger/action attributes by kind.
        if not await element.count():
            element = page.locator('[data-trigger="alarm.item.selection.enter"]').filter(
                has=page.get_by_text(label, exact=True))
        await element.scroll_into_view_if_needed()
        return element

    if path == "cancel_wrong_then_edit":
        await (await row(other)).get_by_role("button").first.click()
        await page.locator('[data-action="alarm.edit.delete"], [data-trigger="alarm.edit.delete"]').wait_for()
        # OS back is the same ordinary exit available to the policy.
        from bench_env.env.base import Action, ActionType
        await env.step(Action(ActionType.BACK, {}))
        assert (await env.get_observation()).state["apps"]["clock"]["alarms"] == alarms

    selected = other if path == "delete_wrong" else target
    element = await row(selected)
    if path == "long_press":
        box = await element.bounding_box()
        await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await page.mouse.down()
        try:
            await page.locator('[data-action="alarm.selection.delete"], [data-trigger="alarm.selection.delete"]').wait_for()
        finally:
            await page.mouse.up()
        await page.get_by_role("button", name="删除", exact=True).click()
    else:
        await element.get_by_role("button").first.click()
        await page.locator('[data-action="alarm.edit.delete"], [data-trigger="alarm.edit.delete"]').click()

    last = await env.get_observation()
    judge = Evaluator(judge_mode="state", eval_mode="text")._evaluate_with_state(
        task, initial, last, SimpleNamespace(agent_answer=None))
    assert judge.passed is (path != "delete_wrong")
    if path == "delete_wrong":
        assert not judge.success and not judge.clean

    full, _ = await Controller.setup(env, make_task("clock.DeleteAlarm", 41), eval_mode="text")
    assert len(full.state["apps"]["clock"]["alarms"]) == 7
    assert await page.evaluate("() => __SIM__.getFeatureAvailability().disabled") == []
