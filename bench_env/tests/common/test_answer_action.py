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

from __future__ import annotations

import shlex
from types import SimpleNamespace

import pytest

from bench_env.agent.generic_v2 import GenericAgentV2
from bench_env.agent.human import HumanAgent
from bench_env.env.base import Action, ActionType, Observation, StepResult
from bench_env.env.stopwatch import StopWatch
from bench_env.env.mobile_gym import MobileGymEnv
from bench_env.env.real_device import RealDeviceEnv
from bench_env.runner.base import Controller, EpisodeResult, Evaluator, ExecutionResult
from bench_env.task.judge import JudgeResult
from bench_env.task.vlm_judge import VLMJudgeOutput


def _make_obs(step_idx: int = 0) -> Observation:
    return Observation(
        screenshot_base64="",
        route={"app": "demo", "path": "/"},
        state={"apps": {}, "os": {}},
        step_idx=step_idx,
    )


@pytest.mark.asyncio
async def test_mobile_gym_answer_action_is_non_terminal_and_skips_delay(monkeypatch: pytest.MonkeyPatch):
    env = MobileGymEnv(url="http://localhost", delay_after_action=9.9, verbose=False)
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def fake_get_observation(*, include_state: bool = True) -> Observation:
        return _make_obs(env._step_count)

    monkeypatch.setattr("bench_env.env.mobile_gym.asyncio.sleep", fake_sleep)
    env._get_observation = fake_get_observation  # type: ignore[method-assign]

    result = await env.step(Action.answer("42"))

    assert result.done is False
    assert result.info == {"action_type": ActionType.ANSWER, "answer": "42"}
    assert env.agent_answer == "42"
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_mobile_gym_escalating_waits_preserve_actions_and_allow_next_answer(monkeypatch):
    # The stalled weather rollout requested this sequence before its first update.
    # Exercise real step dispatch without a browser or physical waiting.
    requested = [3, 5, 8, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600]
    env = MobileGymEnv(url="http://localhost", delay_after_action=0, verbose=False)
    sleeps = []

    async def fake_sleep(seconds):
        assert 0 <= seconds <= 10, "A model WAIT must not hold the batch for minutes"
        sleeps.append(seconds)

    async def fake_observation(*, include_state=True):
        return _make_obs(env._step_count)

    monkeypatch.setattr("bench_env.env.mobile_gym.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(env, "_get_observation", fake_observation)
    for step, seconds in enumerate(requested, 1):
        action = Action(ActionType.WAIT, {"value": seconds})
        result = await env.step(action)
        assert result.observation.step_idx == step and not result.done
        assert action.data == {"value": seconds}  # Record what the policy requested.

    assert sum(sleeps) == 146  # Previously 8,851 seconds, excluding the zero action delay.
    count = len(sleeps)
    result = await env.step(Action.answer("北京"))
    assert result.info == {"action_type": ActionType.ANSWER, "answer": "北京"}
    assert env.agent_answer == "北京" and len(sleeps) == count


@pytest.mark.asyncio
async def test_mobile_gym_reset_clears_agent_answer() -> None:
    env = MobileGymEnv(url="http://localhost", verbose=False)
    env._agent_answer = "old"

    async def fake_reset_sim() -> None:
        return None

    async def fake_wait_ready(*, app_ids=None) -> None:
        return None

    env._reset_sim = fake_reset_sim  # type: ignore[method-assign]
    env._wait_ready = fake_wait_ready  # type: ignore[method-assign]

    await env.reset(app_ids=[])

    assert env.agent_answer is None


@pytest.mark.asyncio
async def test_real_device_answer_action_keeps_latest_answer() -> None:
    env = RealDeviceEnv(delay_after_action=5.0)

    async def fake_get_observation() -> Observation:
        return _make_obs(env._step_count)

    env._get_observation = fake_get_observation  # type: ignore[method-assign]

    first = await env.step(Action.answer("第一次"))
    second = await env.step(Action.answer("第二次"))
    complete = await env.step(Action.complete("完成说明"))

    assert first.done is False
    assert second.info == {"action_type": ActionType.ANSWER, "answer": "第二次"}
    assert complete.done is True
    assert env.agent_answer == "第二次"
    assert env.agent_message == "完成说明"


@pytest.mark.asyncio
async def test_real_device_yadb_quotes_text_for_remote_shell() -> None:
    env = RealDeviceEnv()
    adb_calls: list[tuple[str, ...]] = []

    async def fake_adb(*args: str) -> str:
        adb_calls.append(args)
        return ""

    env._adb = fake_adb  # type: ignore[method-assign]

    text = "This is a benchmark post body"
    success = await env._type_via_yadb(text)

    assert success is True
    assert adb_calls[0][-1] == shlex.quote(text)


class _CaptureTask:
    def __init__(self) -> None:
        self.last_input = None

    def evaluate(self, judge_input):
        self.last_input = judge_input
        return JudgeResult(success=judge_input.answer == "结构化答案", clean=True, progress=1.0)


@pytest.mark.asyncio
async def test_evaluator_uses_agent_answer_instead_of_agent_message() -> None:
    evaluator = Evaluator(judge_mode="state")
    task = _CaptureTask()
    exec_result = ExecutionResult(
        steps=1,
        trace=[],
        runtime_s=0.1,
        finished=True,
        truncated=False,
        stop_reason=ActionType.COMPLETE,
        agent_message="这里有噪声的完成说明",
        agent_answer="结构化答案",
    )

    result = await evaluator.evaluate(task, _make_obs(), _make_obs(1), exec_result)

    assert task.last_input.answer == "结构化答案"
    assert result.success is True


class _GroundedCustomTask:
    answer_fields = [
        {"type": "text", "label": "出发地"},
        {"type": "text", "label": "目的地"},
    ]

    def __init__(self) -> None:
        self.last_answer = None

    def evaluate(self, judge_input):
        self.last_answer = judge_input.answer
        return JudgeResult(
            success=judge_input.answer == "出发地: 上海; 目的地: 广州",
            clean=True,
            progress=1.0,
        )


@pytest.mark.asyncio
async def test_grounded_custom_task_hydrates_labeled_answer_fields() -> None:
    evaluator = Evaluator(judge_mode="state", eval_mode="grounded")
    task = _GroundedCustomTask()
    last_obs = _make_obs(1)
    last_obs.state["apps"]["answer_sheet"] = {
        "fields": task.answer_fields,
        "answers": {"0": "上海", "1": "广州"},
        "submitted": True,
    }
    exec_result = ExecutionResult(
        steps=1,
        trace=[],
        runtime_s=0.1,
        finished=True,
        truncated=False,
        stop_reason=ActionType.COMPLETE,
        agent_message="完成",
        agent_answer="不应使用这条文本答案",
    )

    result = await evaluator.evaluate(task, _make_obs(), last_obs, exec_result)

    assert task.last_answer == "出发地: 上海; 目的地: 广州"
    assert result.success is True


class _FakeVLMJudge:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def evaluate(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return VLMJudgeOutput(
            result=JudgeResult.ok(),
            prompt=[{"role": "system", "content": "ok"}],
            response="{}",
        )


class _FakeEpisode:
    def __init__(self) -> None:
        self.saved: tuple | None = None

    def get_trajectory_for_vlm(self):
        return [{"step": 1, "action_type": "ANSWER", "action_data": {"value": "42"}}]

    def save_vlm_judge(self, prompt, response) -> None:
        self.saved = (prompt, response)


@pytest.mark.asyncio
async def test_evaluator_passes_answer_message_and_stop_reason_to_vlm() -> None:
    vlm_judge = _FakeVLMJudge()
    evaluator = Evaluator(judge_mode="vlm", vlm_judge=vlm_judge)
    episode = _FakeEpisode()
    exec_result = ExecutionResult(
        steps=1,
        trace=[],
        runtime_s=0.1,
        finished=False,
        truncated=True,
        stop_reason="MAX_STEPS",
        agent_message="完成说明",
        agent_answer="最终答案",
    )

    result = await evaluator.evaluate(
        SimpleNamespace(description="测试任务"),
        _make_obs(),
        _make_obs(1),
        exec_result,
        episode,
    )

    assert result.success is True
    kwargs = vlm_judge.calls[0]["kwargs"]
    assert kwargs["agent_answer"] == "最终答案"
    assert kwargs["agent_message"] == "完成说明"
    assert kwargs["stop_reason"] == "MAX_STEPS"
    assert episode.saved is not None


class _AnswerOnlyAgent:
    name = "answer-only"

    def __init__(self, action: Action) -> None:
        self._action = action
        self.history: list = []

    def reset(self, task: str) -> None:
        self.task = task

    def act(self, obs: Observation) -> Action:
        return self._action


class _AnswerTrackingEnv:
    def __init__(self) -> None:
        self._agent_answer: str | None = None
        self._agent_message: str | None = None
        self.stopwatch = StopWatch()

    async def get_state(self, *, required_apps: list[str] | None = None) -> dict:
        return {}

    async def step(self, action: Action) -> StepResult:
        if action.action_type == ActionType.ANSWER:
            self._agent_answer = action.data["value"]
            return StepResult(observation=_make_obs(1), done=False, info={"action_type": ActionType.ANSWER})
        raise AssertionError(f"unexpected action: {action.action_type}")

    @property
    def agent_answer(self) -> str | None:
        return self._agent_answer

    @property
    def agent_message(self) -> str | None:
        return self._agent_message


class _TaskForController:
    id = "demo.AnswerTask"
    description = "提交一个答案"
    suite = "demo"
    apps: list[str] = []

    def teardown(self, env) -> None:
        return None


@pytest.mark.asyncio
async def test_controller_run_keeps_answer_nonterminal_for_text_non_answer_task() -> None:
    env = _AnswerTrackingEnv()
    agent = _AnswerOnlyAgent(Action.answer("最后答案"))
    task = _TaskForController()

    exec_result, *_ = await Controller.run(
        env,
        agent,
        task,
        _make_obs(),
        max_steps=1,
        recorder=None,
        eval_mode="text",
    )

    assert exec_result.truncated is True
    assert exec_result.stop_reason == "MAX_STEPS"
    assert exec_result.agent_answer == "最后答案"
    assert exec_result.agent_message is None


class _DirectAnswerTask(_TaskForController):
    answer_fields = [{"type": "text", "label": "答案"}]


class _SetupDirectAnswerTask(_DirectAnswerTask):
    task_name = None
    params = {"demo": 1}

    @property
    def description(self) -> str:
        return self.task_name or "查找并回答问题"

    async def setup(self, env) -> Observation:
        return _make_obs()


@pytest.mark.asyncio
async def test_controller_setup_adds_minimal_text_answer_hint() -> None:
    task = _SetupDirectAnswerTask()

    initial_obs, params = await Controller.setup(None, task, eval_mode="text")

    assert initial_obs == _make_obs()
    assert params == {"demo": 1}
    assert task.description == "查找并回答问题 完成必要操作后，用 ANSWER 提交最终答案；提交后任务结束"


@pytest.mark.asyncio
async def test_controller_run_terminates_on_answer_for_text_answer_task() -> None:
    env = _AnswerTrackingEnv()
    agent = _AnswerOnlyAgent(Action.answer("最后答案"))

    exec_result, *_ = await Controller.run(
        env,
        agent,
        _DirectAnswerTask(),
        _make_obs(),
        max_steps=5,
        recorder=None,
        eval_mode="text",
    )

    assert exec_result.finished is True
    assert exec_result.truncated is False
    assert exec_result.stop_reason == ActionType.COMPLETE
    assert exec_result.steps == 1
    assert exec_result.agent_answer == "最后答案"


@pytest.mark.asyncio
async def test_controller_run_keeps_answer_nonterminal_for_grounded_answer_task() -> None:
    env = _AnswerTrackingEnv()
    agent = _AnswerOnlyAgent(Action.answer("最后答案"))

    exec_result, *_ = await Controller.run(
        env,
        agent,
        _DirectAnswerTask(),
        _make_obs(),
        max_steps=1,
        recorder=None,
        eval_mode="grounded",
    )

    assert exec_result.finished is False
    assert exec_result.truncated is True
    assert exec_result.stop_reason == "MAX_STEPS"


def test_episode_result_abort_forces_failure_even_if_judge_passes() -> None:
    result = EpisodeResult(
        task_id="demo.AnswerTask",
        task_name="demo",
        suite="demo",
        execution=ExecutionResult(
            steps=1,
            trace=[],
            runtime_s=0.1,
            finished=True,
            truncated=False,
            stop_reason=ActionType.ABORT,
            agent_message="放弃",
            agent_answer="正确答案",
        ),
        judge=JudgeResult(success=True, clean=True, progress=1.0),
    )

    assert result.goal_success is True
    assert result.success is False


def test_generic_v2_parse_answer_action() -> None:
    agent = GenericAgentV2(llm=object())

    action = agent.parse_response(
        "<think>先回答再继续</think><answer>{\"action\": \"ANSWER\", \"value\": \"42\"}</answer>"
    )

    assert action.action_type == ActionType.ANSWER
    assert action.data == {"value": "42"}


def test_generic_v2_prompt_explains_answer_and_termination_semantics() -> None:
    prompt = GenericAgentV2.SYSTEM_PROMPT

    assert "若任务要求通过答题卡 APP 回答" in prompt
    assert "若要求直接回答，最后使用 ANSWER" in prompt
    assert "ANSWER 会结束直接回答任务" in prompt
    assert "其他任务完成后使用 COMPLETE" in prompt


def test_generic_v2_prompt_explains_type_focus_semantics() -> None:
    prompt = GenericAgentV2.SYSTEM_PROMPT

    assert '{"action": "TYPE", "value": "文本内容", "point": [x, y]}' in prompt
    assert '默认用 "point" 指定输入框' in prompt
    assert "仅当输入框已明确聚焦时才可省略" in prompt
    assert "否则必须先 CLICK" in prompt


def test_human_agent_supports_answer_command(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = HumanAgent()
    agent.reset("回答问题")
    monkeypatch.setattr("builtins.input", lambda _: "a 结构化答案")

    action = agent.act(_make_obs())

    assert action.action_type == ActionType.ANSWER
    assert action.data == {"value": "结构化答案"}
