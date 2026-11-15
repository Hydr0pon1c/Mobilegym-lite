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

import asyncio
import base64
import copy
import io
import json
import threading
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest
from PIL import Image

from bench_env.env.base import Action, Observation, StepResult
from bench_env.env.recorder import EpisodeRecorder, RunRecorder
from bench_env.env.stopwatch import StopWatch
from bench_env.runner.base import Controller, _record_step_async


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 3, 27, 1, 10, 0, tzinfo=tz)


def test_start_run_uses_unique_directory_when_timestamp_collides(tmp_path, monkeypatch):
    import bench_env.env.recorder as recorder_module

    monkeypatch.setattr(recorder_module, "datetime", _FixedDateTime)

    rec1 = RunRecorder(tmp_path, save_trajectory=False)
    run_dir_1 = rec1.start_run(agent="a", model_name="m")
    rec1.finish_run()

    rec2 = RunRecorder(tmp_path, save_trajectory=False)
    run_dir_2 = rec2.start_run(agent="a", model_name="m")
    rec2.finish_run()

    assert run_dir_1 != run_dir_2
    assert run_dir_1.name == "20260327_011000"
    assert run_dir_2.name.startswith("20260327_011000")


@pytest.fixture
def recording_inputs():
    image = io.BytesIO()
    Image.new("RGB", (32, 64), "white").save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode()
    obs = Observation(screenshot_base64=encoded, route={"app": "demo", "path": "/"}, state={})
    prompt = [{"role": "user", "content": [
        {"type": "text", "text": "点击按钮"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]}]
    return dict(step_idx=1, obs=obs, action=Action.click([500, 500], thought="点击"),
                route=obs.route, model_response='{"action": "CLICK"}', model_prompt=prompt)


@pytest.mark.asyncio
async def test_threaded_recording_preserves_artifacts_and_inputs(tmp_path, recording_inputs):
    original = copy.deepcopy(recording_inputs)
    directories = []
    for name in ("sync", "thread"):
        directory = tmp_path / name
        directory.mkdir()
        episode = EpisodeRecorder(RunRecorder(directory), "demo.task", "demo", directory)
        for step in (1, 2):
            kwargs = {**recording_inputs, "step_idx": step}
            if name == "sync":
                episode.record_step(**kwargs)
            else:
                await _record_step_async(episode, **kwargs)
        episode.finish({})
        directories.append(directory)
    artifacts = [{p.name: p.read_bytes() for p in d.iterdir()} for d in directories]
    assert artifacts[0] == artifacts[1]
    assert {"trajectory.json", "step_001.png", "step_002_annot.jpg",
            "step_002_prompt.json", "step_002_response.txt"} <= artifacts[1].keys()
    assert recording_inputs == original


def _controller_case(recorder, inputs, task_id="demo.task"):
    events = []
    obs = inputs["obs"]
    agent = SimpleNamespace(name="test", history=[], reset=lambda _: None)
    env = SimpleNamespace(stopwatch=StopWatch(), agent_message=None, agent_answer=None)

    def act(current):
        agent.history.append(SimpleNamespace(llm_prompt=inputs["model_prompt"]))
        return inputs["action"] if current.step_idx == 0 else Action.complete()

    async def step(action):
        events.append("action")
        return StepResult(observation=replace(obs, step_idx=1), done=action.is_terminal,
                          info={"stop_reason": action.action_type if action.is_terminal else None})

    async def get_state(**kwargs):
        return obs.state

    agent.act = act
    env.step = step
    env.get_state = get_state
    task = SimpleNamespace(id=task_id, description="demo", apps=[],
                           teardown=lambda _: events.append("teardown"))
    return Controller.run(env, agent, task, obs, max_steps=2, recorder=recorder), events


@pytest.mark.asyncio
async def test_recording_allows_other_episodes_to_progress(tmp_path, monkeypatch, recording_inputs):
    recorder = RunRecorder(tmp_path)
    recorder.start_run()
    barrier = threading.Barrier(2, timeout=5)
    main_thread = threading.get_ident()
    real_record = EpisodeRecorder.record_step
    calls = {}

    def record(episode, **kwargs):
        assert threading.get_ident() != main_thread
        # Both episodes must enter recording together; a blocked event loop
        # cannot reach the second write and will break this barrier.
        barrier.wait()
        real_record(episode, **kwargs)
        calls[episode.task_id].append("record")

    monkeypatch.setattr(EpisodeRecorder, "record_step", record)
    runners = []
    for task_id in ("demo.first", "demo.second"):
        run, events = _controller_case(recorder, recording_inputs, task_id)
        runners.append(run)
        calls[task_id] = events
    results = await asyncio.wait_for(asyncio.gather(*runners), timeout=10)
    for execution, _, _, episode, _ in results:
        assert execution.error is None
        assert execution.finished and execution.steps == 2
        assert calls[episode.task_id] == ["record", "action", "record", "action", "teardown"]
        episode.finish({})
        trajectory = json.loads((episode._episode_dir / "trajectory.json").read_text())
        assert [s["step"] for s in trajectory] == [1, 2]
    recorder.finish_run()


@pytest.mark.asyncio
async def test_recording_failure_still_terminates_episode(tmp_path, monkeypatch, recording_inputs):
    recorder = RunRecorder(tmp_path)
    recorder.start_run()

    def fail(*args, **kwargs):
        raise OSError("recording failed")

    monkeypatch.setattr(EpisodeRecorder, "record_step", fail)
    run, events = _controller_case(recorder, recording_inputs)
    execution, _, _, episode, _ = await run
    assert execution.error == "OSError: recording failed"
    assert execution.stop_reason == "ERROR"
    assert not execution.finished
    assert events == ["teardown"]
    episode.finish({})
    recorder.finish_run()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_write", [False, True])
async def test_cancel_waits_for_recording_before_teardown(tmp_path, monkeypatch, recording_inputs, fail_write):
    recorder = RunRecorder(tmp_path)
    recorder.start_run()
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    finished = threading.Event()
    real_record = EpisodeRecorder.record_step

    def blocked_record(episode, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        try:
            if fail_write:
                raise OSError("recording failed during cancellation")
            real_record(episode, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(EpisodeRecorder, "record_step", blocked_record)
    run, events = _controller_case(recorder, recording_inputs)
    task = asyncio.create_task(run)
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not events  # No action/teardown while the worker owns inputs.
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        recorder.finish_run()
    assert finished.is_set()
    assert events == ["teardown"]
