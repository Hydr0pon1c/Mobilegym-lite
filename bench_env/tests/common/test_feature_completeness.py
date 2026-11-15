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

"""Task scope, dependency protection and episode integration without a model."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import random

import pytest

from bench_env.completeness import apply_completeness
from bench_env.feature_completeness import MIX_MODE, MIX_PERIOD, MODES, PROFILES, PROFILE_TASKS, TASK_IDS, VERSION, LAYOUT_MODE, LAYOUT_VERSION, SELECTIVE_MODE, feature_plan
from bench_env.task.registry import TaskRegistry


def task(tid, **kwargs):
    return SimpleNamespace(id=tid, **kwargs)


def test_reviewed_scope_and_membership():
    registry = TaskRegistry()
    train = Path(__file__).resolve().parents[2] / 'splits/env_effect_v3_train.txt'
    expected = {tid for tid in train.read_text().splitlines()
                if tid and not tid.startswith('#') and
                set(registry.create_task(tid).apps) & {'clock', 'notes', 'calendar', 'wechat'}}
    assert TASK_IDS == expected and len(TASK_IDS) == 78
    assert sum(map(len, PROFILE_TASKS.values())) == 85
    assert PROFILES.keys() == PROFILE_TASKS.keys()
    for tid in TASK_IDS:
        apps = registry.create_task(tid).apps
        assigned = [name.split('_')[0] for name, ids in PROFILE_TASKS.items() if tid in ids]
        assert len(assigned) == len(set(assigned))
        assert set(assigned) <= set(apps)


def test_plus_extends_only_reviewed_app_profiles_and_retains_selective_restorations():
    from bench_env.feature_completeness import PLUS_MODE, PLUS_PROFILES, PLUS_PROFILE_TASKS
    registry = TaskRegistry()
    train = Path(__file__).resolve().parents[2] / 'splits/env_effect_v3_train.txt'
    ids = {line for line in train.read_text().splitlines() if line and not line.startswith('#')}
    apps = {'bilibili_commerce': 'bilibili', 'spotify_premium': 'spotify', 'alipay_video': 'alipay'}
    for profile, app in apps.items():
        assert PLUS_PROFILE_TASKS[profile] == {tid for tid in ids if app in registry.create_task(tid).apps}
    for tid in ids:
        before = feature_plan(task(tid), SELECTIVE_MODE)
        after = feature_plan(task(tid), PLUS_MODE)
        extra = {module for name, members in PLUS_PROFILE_TASKS.items() if tid in members
                 for module in PLUS_PROFILES[name]}
        assert set(after['disabled']) == set(before['disabled']) | extra
        assert after['kept'] == before['kept']
        assert after == feature_plan(task(tid + '_i7', params={'arbitrary': 'value'}), PLUS_MODE)
    for tid in ('bilibili.FutureTask', 'spotify.FutureTask', 'alipay.FutureTask'):
        assert not feature_plan(task(tid), PLUS_MODE)['supported']
        assert not feature_plan(task(tid), PLUS_MODE)['disabled']


def test_plus_keeps_crossapp_required_functions_and_resource_routes():
    from bench_env.feature_completeness import PLUS_MODE
    social = feature_plan(task('crossapp_content.BilibiliTripleLikeThenMoments'), PLUS_MODE)
    assert set(social['disabled']) == {'bilibili.vip', 'bilibili.shop', 'wechat.profile_edit'}
    music = feature_plan(task('crossapp_content.SpotifySaveCurrentSongToNotes'), PLUS_MODE)
    assert set(music['disabled']) == {'spotify.premium', 'notes.todo'}
    finance = feature_plan(task('crossapp_commerce.AlipayMonthlyToNotesAndWechat'), PLUS_MODE)
    assert set(finance['disabled']) == {'alipay.video', 'notes.todo', 'wechat.moments'}


def test_nested_reproducible_masks_do_not_consult_params_or_rng():
    rng = random.getstate()
    for tid in TASK_IDS:
        zero = feature_plan(task(tid), 'feature0')
        one = feature_plan(task(tid), 'feature1')
        assert zero['effective'] and not zero['kept']
        assert set(one['disabled']) <= set(zero['disabled'])
        assert len(one['kept']) == (len(zero['candidates']) + 1) // 2
        assert set(one['kept']).isdisjoint(one['disabled'])
        assert set(one['kept'] + one['disabled']) == set(zero['candidates'])
        assert feature_plan(task(tid + '_i9', params={'branch': 'other'}), 'feature1') == one
    assert random.getstate() == rng
    for mode in MODES:
        unknown = feature_plan(task('wechat.FutureTask'), mode)
        assert not unknown['supported'] and not unknown['disabled']
        assert unknown['remaining_fraction'] is None


@pytest.mark.parametrize('tid,protected', [
    ('clock.CompareCityTimeDiff', 'clock.world'),  # reference never opens world clock
    ('notes.CreateNoteWithReminder', 'notes.settings'),
    ('calendar.DateCalcForward', 'calendar.date_calculate'),
    ('calendar.CalculateDateInterval', 'calendar.date_calculate'),
    ('wechat.ToggleWechatSports', 'wechat.profile_edit'),
    ('wechat.SetFriendChatOnly', 'wechat.moments'),
    ('crossapp_content.BilibiliTripleLikeThenMoments', 'wechat.moments'),
])
def test_semantic_dependencies_beyond_observed_reference_routes(tid, protected):
    assert protected not in feature_plan(task(tid), 'feature0')['disabled']


def test_conditional_tasks_keep_the_unvisited_app_branch():
    for tid, profile in [
        ('crossapp_life.RealisticTrip001', 'notes_note'),
        ('crossapp_life.CalendarFreeWeatherInvite', 'wechat_communication'),
        ('crossapp_life.RestaurantRatingInviteCalendar', 'calendar_dates'),
    ]:
        assert profile in feature_plan(task(tid), 'feature0')['profiles']


def test_selective_scope_is_bounded_and_independent_of_sampling():
    from bench_env.splits import resolve_split
    changed = effective = assignments = 0
    rng = random.getstate()
    for tid in resolve_split('env_effect_v3_train'):
        original = feature_plan(task(tid), 'feature0')
        candidate = feature_plan(task(tid), SELECTIVE_MODE)
        assert candidate['profiles'] == original['profiles']
        assert candidate['candidates'] == original['candidates']
        assert set(candidate['disabled']) <= set(original['disabled'])
        assert set(candidate['kept']) <= {'clock.world', 'calendar.settings', 'wechat.profile_edit'}
        assert set(candidate['kept'] + candidate['disabled']) == set(original['disabled'])
        changed += candidate['disabled'] != original['disabled']
        effective += candidate['effective']
        assignments += len(candidate['disabled'])
        for seed in (41, 42, 43):
            assert feature_plan(task(tid + '_i9', params={'branch': seed}, _seed=seed),
                                SELECTIVE_MODE, rollout_index=seed) == candidate
    assert (changed, effective, assignments) == (36, 72, 87)
    assert random.getstate() == rng


@pytest.mark.parametrize('tid,kept,disabled', [
    ('clock.ToggleAlarm', ['clock.world'], ['clock.stopwatch', 'clock.timer']),
    ('calendar.DateCalcForward', ['calendar.settings'], []),
    ('wechat.ReadContactRegion', ['wechat.profile_edit'], ['wechat.moments']),
    ('crossapp_life.CalendarFreeWeatherInvite', ['calendar.settings', 'wechat.profile_edit'], ['wechat.moments']),
    ('crossapp_life.RealisticTrip001', ['wechat.profile_edit'], ['notes.todo', 'wechat.moments']),
    ('crossapp_work.WeatherConditionalCancelMeeting', ['clock.world'], ['clock.stopwatch', 'clock.timer']),
    ('calendar.CreateEvent', [], ['calendar.date_calculate']),
    ('wechat.PostMomentsText', [], ['wechat.profile_edit']),
    ('notes.PinNote', [], ['notes.todo']),
])
def test_selective_restores_functions_only_in_the_reviewed_profiles(tid, kept, disabled):
    record = feature_plan(task(tid), SELECTIVE_MODE)
    assert record['kept'] == kept
    assert record['disabled'] == disabled
    assert 'layout' not in record and 'mixture' not in record


@pytest.mark.asyncio
async def test_feature_injection_does_not_write_app_data_and_full_clears_mask():
    t = task('notes.CreateNewNote')
    obs = object()
    env = SimpleNamespace(page=SimpleNamespace(evaluate=AsyncMock(return_value={
        'version': VERSION, 'disabled': ['notes.todo']})),
        get_observation=AsyncMock(return_value=obs), set_state=AsyncMock())
    assert await apply_completeness(env, t, obs, 'feature0') is obs
    env.set_state.assert_not_called()
    assert t._completeness_record['disabled'] == ['notes.todo']
    assert await apply_completeness(env, t, obs, None) is obs
    assert t._completeness_record is None
    assert 'setDisabledFeatures?.([])' in env.page.evaluate.call_args.args[0]


@pytest.mark.asyncio
async def test_ignored_or_old_frontend_is_an_error():
    env = SimpleNamespace(page=SimpleNamespace(evaluate=AsyncMock(return_value=None)))
    with pytest.raises(ValueError, match='readback mismatch'):
        await apply_completeness(env, task('notes.CreateNewNote'), None, 'feature0')


def test_layout_keeps_feature0_masks_and_covers_all_78_supported_templates():
    from bench_env.splits import resolve_split
    affected = set()
    for tid in resolve_split('env_effect_v3_train'):
        baseline = feature_plan(task(tid), 'feature0')
        candidate = feature_plan(task(tid), LAYOUT_MODE)
        layout = candidate.pop('layout')
        candidate['mode'] = 'feature0'
        assert candidate == baseline
        if layout['effective']:
            affected.add(tid)
        assert layout['effective'] is bool(layout['modules'])
    assert affected == TASK_IDS
    assert len(affected) == 78


@pytest.mark.asyncio
@pytest.mark.parametrize('actual', [
    None,
    {'version': VERSION, 'disabled': ['notes.todo']},
    {'version': VERSION, 'disabled': ['notes.todo'], 'layout': {'version': LAYOUT_VERSION, 'mode': 'compact'}},
])
async def test_layout_rejects_missing_or_ignored_capability(actual):
    env = SimpleNamespace(page=SimpleNamespace(evaluate=AsyncMock(return_value=actual)))
    with pytest.raises(ValueError, match='readback mismatch'):
        await apply_completeness(env, task('notes.CreateNewNote'), None, LAYOUT_MODE)


@pytest.mark.asyncio
async def test_layout_is_recorded_without_touching_app_data():
    t = task('notes.CreateNewNote')
    obs = object()
    env = SimpleNamespace(page=SimpleNamespace(evaluate=AsyncMock(return_value={
        'version': VERSION, 'disabled': ['notes.todo'],
        'layout': {'version': LAYOUT_VERSION, 'mode': 'preserve'}})),
        get_observation=AsyncMock(return_value=obs), set_state=AsyncMock())
    assert await apply_completeness(env, t, obs, LAYOUT_MODE) is obs
    env.set_state.assert_not_called()
    assert t._completeness_record['layout']['modules'] == ['notes.todo']
    assert 'mixture' not in t._completeness_record


def test_mixture_keeps_seven_feature0_rollouts_and_one_full_per_template():
    from bench_env.splits import resolve_split
    rng = random.getstate()
    pruned = full_supported = full_unsupported = 0
    for tid in resolve_split('env_effect_v3_train'):
        zero = feature_plan(task(tid), 'feature0')
        plans = [feature_plan(task(tid), MIX_MODE, rollout_index=i) for i in range(MIX_PERIOD)]
        if tid in TASK_IDS:
            assert sum(p['effective'] for p in plans) == 7
        else:
            assert all(not p['effective'] and not p['supported'] for p in plans)
        for i, p in enumerate(plans):
            pruned += p['effective']
            full_supported += p['supported'] and not p['effective']
            full_unsupported += not p['supported']
            assert p['disabled'] == (zero['disabled'] if p['effective'] else [])
            assert set(p['kept'] + p['disabled']) == set(zero['candidates'])
            assert feature_plan(task(tid + '_i9', params={'answer': 'changed'}, _seed=999),
                                MIX_MODE, rollout_index=i) == p
            assert feature_plan(task(tid), MIX_MODE, rollout_index=i + MIX_PERIOD)['disabled'] == p['disabled']
    assert (pruned, full_supported, full_unsupported) == (546, 78, 912)
    assert random.getstate() == rng


@pytest.mark.parametrize('index', [-1, None, True, 1.5, '1'])
def test_mixture_rejects_ambiguous_rollout_index(index):
    with pytest.raises(ValueError, match='rollout index'):
        feature_plan(task('notes.CreateNewNote'), MIX_MODE, rollout_index=index)


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit,expected', [(None, 9), (2, 2)])
async def test_runner_distinguishes_recording_trial_from_completeness_rollout(monkeypatch, explicit, expected):
    from bench_env.runner.base import Controller
    setup = AsyncMock(return_value=(object(), {}))
    run = AsyncMock(return_value=object())
    monkeypatch.setattr(Controller, 'setup', setup)
    monkeypatch.setattr(Controller, 'run', run)
    await Controller.run_loop(object(), object(), object(), trial_id=9,
                              completeness=MIX_MODE, completeness_rollout_index=explicit)
    assert setup.call_args.kwargs['completeness_rollout_index'] == expected
    assert run.call_args.args[6] == 9
