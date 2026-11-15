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

"""Browser gates and pool isolation. Run with -m live --sim-url <new build>."""
import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from bench_env.env.pool import EnvPool

pytestmark = [pytest.mark.live, pytest.mark.asyncio(loop_scope="session")]


async def open_route(env, app, path, expected=None):
    await env.page.evaluate("([app, path]) => __OS__.openApp(app, path)", [app, path])
    try:
        await env.page.wait_for_function("([app, path]) => { const r = __OS__.getAppRoute(); return r?.app === app && r.path === path; }",
                                         arg=[app, path if expected is None else expected])
    except PlaywrightTimeoutError:
        raise AssertionError(f"Opening {app}:{path}, expected {expected or path}, got {await env.get_route()}") from None


async def test_all_modules_reject_deep_links_and_restore_full(env):
    await env.reset()
    cases = {
        'clock.world': ['/world'], 'clock.stopwatch': ['/stopwatch'], 'clock.timer': ['/timer'],
        'notes.todo': ['/todo'], 'notes.settings': ['/settings'],
        'calendar.date_calculate': ['/date-calculate'], 'calendar.settings': ['/settings'],
        'wechat.moments': ['/moments', '/moments/self', '/moments/media-picker',
                           '/post-moment', '/post-text-moment', '/select-location', '/me/moments-album'],
        'wechat.profile_edit': ['/my-profile-detail', '/me/signature', '/me/pat', '/me/address/add'],
    }
    for module, paths in cases.items():
        app = module.split('.')[0]
        home = '/alarm' if app == 'clock' else '/'
        await env.page.evaluate('(ids) => __SIM__.setDisabledFeatures(ids)', [module])
        for path in paths:
            await open_route(env, app, path, home)
        await env.page.evaluate('() => __SIM__.setDisabledFeatures([])')
        await open_route(env, app, paths[0])
    await env.reset()
    assert await env.page.evaluate('() => __SIM__.getFeatureAvailability().disabled') == []


async def test_entries_and_shared_wechat_paths(env):
    await env.reset()
    await env.page.evaluate("() => __SIM__.setDisabledFeatures(['wechat.moments', 'wechat.profile_edit'])")
    for path, selectors in [
        ('/discover', ['[data-trigger="discover.moments.open"]']),
        ('/me', ['[data-trigger="me.moments.open"]', '[data-trigger="profile.detail.open"]']),
        ('/settings', ['[data-trigger="profile.detail.open"]']),
    ]:
        await open_route(env, 'wechat', path)
        for selector in selectors:
            assert await env.page.locator(selector + ':visible').count() == 0
    wxid = await env.page.evaluate('() => __SIM__.getState().apps.wechat.contacts[0].wxid')
    await open_route(env, 'wechat', f'/user-profile/{wxid}')
    assert await env.page.locator('[data-trigger="moments.user.open"]:visible').count() == 0
    # The picker normalizes its URL in a mount effect; wait for that canonical
    # URL rather than racing the transient route before ?albumId=all is added.
    await open_route(env, 'wechat', f'/chat/{wxid}/media-picker', f'/chat/{wxid}/media-picker?albumId=all')
    for path in ['/me/qrcode', '/settings/general/accessibility/wechatSports']:
        await open_route(env, 'wechat', path)
    # Sports remains reachable from search as well as Me/settings.
    await open_route(env, 'wechat', '/search')
    await env.page.locator('input:visible').fill('微信运动')
    await env.page.locator('[data-trigger^="discover.search.feature.wechatSports."]:visible').wait_for()


async def test_clock_and_notes_tabs_and_settings_entries(env):
    await env.reset()
    await env.page.evaluate("() => __SIM__.setDisabledFeatures(['clock.world', 'clock.stopwatch', 'clock.timer', 'notes.todo', 'notes.settings'])")
    await open_route(env, 'clock', '/alarm')
    for label in ['世界时钟', '秒表', '计时器']:
        assert await env.page.get_by_role('button', name=label, exact=True).count() == 0
    await open_route(env, 'notes', '/')
    assert await env.page.locator('[data-visual-app=notes]').get_by_role('button', name='待办', exact=True).count() == 0
    assert await env.page.locator('[data-visual-app=notes]').get_by_role('button', name='设置', exact=True).count() == 0
    await env.page.evaluate("() => __SIM__.setDisabledFeatures(['notes.settings'])")
    await open_route(env, 'notes', '/todo')
    assert await env.page.locator('[data-visual-app=notes]').get_by_role('button', name='设置', exact=True).count() == 0


async def test_calendar_menu_and_retained_calculator(env):
    await env.reset()
    await env.page.evaluate("() => __SIM__.setDisabledFeatures(['calendar.date_calculate'])")
    await open_route(env, 'calendar', '/')
    app = env.page.locator('[data-visual-app=calendar]')
    await app.get_by_role('button', name='更多', exact=True).click()
    assert await app.get_by_role('button', name='日期推算', exact=True).count() == 0
    assert await app.get_by_role('button', name='设置', exact=True).count() == 1
    await app.get_by_role('button', name='取消', exact=True).click()
    await env.page.evaluate("() => __SIM__.setDisabledFeatures(['calendar.settings'])")
    await app.get_by_role('button', name='更多', exact=True).click()
    assert await app.get_by_role('button', name='设置', exact=True).count() == 0
    await app.get_by_role('button', name='日期推算', exact=True).click()
    await app.locator('input[type=date]').fill('2026-09-17')
    await app.locator('input[type=number]').fill('3')
    await app.get_by_role('button', name='开始推算', exact=True).click()
    await app.get_by_text('2026-09-20（9月20日）', exact=True).wait_for()


async def test_page_isolation_and_reuse_reset(request):
    async with EnvPool(request.config.getoption('--sim-url'), n=2, num_browsers=1,
                       delay_after_action=0.1, verbose=False) as pool:
        left, right = pool[0].page, pool[1].page
        await left.evaluate("() => __SIM__.setDisabledFeatures(['notes.todo'])")
        assert await right.evaluate('() => __SIM__.getFeatureAvailability().disabled') == []
        await left.evaluate('() => __SIM__.resetState()')
        assert await left.evaluate('() => __SIM__.getFeatureAvailability().disabled') == []
        await left.evaluate("() => __SIM__.setDisabledFeatures(['clock.timer'])")
        await left.reload()
        await pool[0].wait_ready()
        assert await left.evaluate('() => __SIM__.getFeatureAvailability().disabled') == []


async def test_mix_restores_full_tabs_without_projecting_resources(env):
    from bench_env.completeness import apply_completeness
    from bench_env.completeness_check import make_task
    from bench_env.feature_completeness import MIX_MODE, feature_plan
    from bench_env.runner.base import Controller

    task = make_task('clock.DeleteAlarm', 41)
    initial, _ = await Controller.setup(env, task, eval_mode='text')
    full_index = feature_plan(task, MIX_MODE)['mixture']['full_slot']
    pruned_index = (full_index + 1) % 8
    for index, expected_tabs in [(pruned_index, 0), (full_index, 1), (pruned_index, 0)]:
        obs = await apply_completeness(env, task, initial, MIX_MODE, rollout_index=index)
        assert obs.state['apps'] == initial.state['apps']
        assert len(obs.state['apps']['clock']['alarms']) == 7
        await open_route(env, 'clock', '/alarm')
        assert await env.page.get_by_role('button', name='世界时钟', exact=True).count() == expected_tabs
        assert task._completeness_record['mixture']['rollout_index'] == index


@pytest.mark.parametrize('tid,app,path', [
    ('clock.ToggleAlarm', 'clock', '/world'),
    ('calendar.DateCalcForward', 'calendar', '/settings'),
    ('wechat.ReadContactRegion', 'wechat', '/my-profile-detail'),
    ('crossapp_life.CalendarFreeWeatherInvite', 'calendar', '/settings'),
    ('crossapp_life.CalendarFreeWeatherInvite', 'wechat', '/my-profile-detail'),
    ('crossapp_life.RealisticTrip001', 'wechat', '/my-profile-detail'),
    ('crossapp_work.WeatherConditionalCancelMeeting', 'clock', '/world'),
])
async def test_selective_restores_routes_without_projecting_state(env, tid, app, path):
    from bench_env.completeness import apply_completeness
    from bench_env.completeness_check import make_task
    from bench_env.feature_completeness import SELECTIVE_MODE
    from bench_env.runner.base import Controller

    task = make_task(tid, 41)
    initial, _ = await Controller.setup(env, task, eval_mode='text')
    for mode in ('feature0', SELECTIVE_MODE, 'feature0', SELECTIVE_MODE):
        obs = await apply_completeness(env, task, initial, mode)
        assert obs.state['apps'] == initial.state['apps']
        assert await env.page.evaluate('() => __SIM__.getFeatureLayout().mode') == 'compact'
        await open_route(env, app, path, path if mode == SELECTIVE_MODE else '/alarm' if app == 'clock' else '/')
        assert await env.page.locator('[data-feature-layout-slot]').count() == 0
        # Return before changing gates, matching the normal episode setup lifecycle.
        await open_route(env, app, '/alarm' if app == 'clock' else '/')
    await env.reset()
    assert await env.page.evaluate('() => __SIM__.getFeatureAvailability().disabled') == []


async def test_selective_restored_functions_are_usable(env):
    from bench_env.completeness_check import make_task
    from bench_env.env.base import Action, ActionType
    from bench_env.feature_completeness import SELECTIVE_MODE
    from bench_env.runner.base import Controller

    await Controller.setup(env, make_task('clock.ToggleAlarm', 41), eval_mode='text', completeness=SELECTIVE_MODE)
    await open_route(env, 'clock', '/alarm')
    await env.page.get_by_role('button', name='世界时钟', exact=True).click()
    await env.page.wait_for_function("() => __OS__.getAppRoute()?.path === '/world'")
    await env.page.locator('button[aria-label="添加"]:visible').click()
    await env.page.locator('input:visible').wait_for()
    await env.step(Action(ActionType.BACK, {}))
    assert await env.get_route() == {'app': 'clock', 'path': '/world'}
    for label in ('秒表', '计时器'):
        assert await env.page.get_by_role('button', name=label, exact=True).count() == 0

    await Controller.setup(env, make_task('calendar.DateCalcForward', 41), eval_mode='text', completeness=SELECTIVE_MODE)
    await open_route(env, 'calendar', '/')
    app = env.page.locator('[data-visual-app=calendar]')
    await app.get_by_role('button', name='更多', exact=True).click()
    await app.get_by_role('button', name='设置', exact=True).click()
    before = await env.page.evaluate('() => __SIM__.getState().apps.calendar.settings.showWeekNumber')
    await app.get_by_text('显示周数', exact=True).click()
    assert await env.page.evaluate('() => __SIM__.getState().apps.calendar.settings.showWeekNumber') is not before
    await app.get_by_text('显示周数', exact=True).click()
    await env.step(Action(ActionType.BACK, {}))
    assert await env.get_route() == {'app': 'calendar', 'path': '/'}

    await Controller.setup(env, make_task('wechat.ReadContactRegion', 41), eval_mode='text', completeness=SELECTIVE_MODE)
    await open_route(env, 'wechat', '/me')
    await env.page.locator('[data-trigger="profile.detail.open"]:visible').click()
    await env.page.locator('[data-trigger="profile.signature.open"]:visible').click()
    await env.page.locator('input:visible').fill('selective functional check')
    await env.page.locator('[data-action="profile.signature.submit"]:visible').click()
    await env.page.wait_for_function("() => __OS__.getAppRoute()?.path === '/my-profile-detail'")
    assert await env.page.evaluate('() => __SIM__.getState().apps.wechat.user.signature') == 'selective functional check'
    assert await env.get_route() == {'app': 'wechat', 'path': '/my-profile-detail'}
    await env.step(Action(ActionType.BACK, {}))
    await env.page.wait_for_function("() => __OS__.getAppRoute()?.path === '/me'")
    assert await env.get_route() == {'app': 'wechat', 'path': '/me'}
    assert await env.page.locator('[data-trigger="me.moments.open"]:visible').count() == 0
    await env.reset()


@pytest.mark.parametrize('app,path,mask,control', [
    ('clock', '/alarm', ['clock.world', 'clock.stopwatch', 'clock.timer'], 'button:has-text("闹钟")'),
    ('clock', '/world', ['clock.stopwatch', 'clock.timer'], 'button:has-text("世界时钟")'),
    ('notes', '/', ['notes.todo'], 'button[aria-current="page"]'),
    ('notes', '/', ['notes.settings'], 'button[aria-label="文件夹"]'),
    ('notes', '/todo', ['notes.settings'], 'input'),
    ('calendar', '/', ['calendar.date_calculate'], 'button:has-text("日期跳转")'),
    ('calendar', '/', ['calendar.settings'], 'button:has-text("日期跳转")'),
    ('wechat', '/discover', ['wechat.moments'], '[data-trigger="scan.open"]'),
    ('wechat', '/me', ['wechat.moments'], '[data-trigger="me.settings.open"]'),
    ('wechat', '/me', ['wechat.profile_edit'], '[data-trigger="me.qrcode.open"]'),
    ('wechat', '/settings', ['wechat.profile_edit'], '[data-trigger="settings.security.open"]'),
    ('wechat', '/user-profile/{contact}', ['wechat.moments'], '[data-trigger="chat.open"]'),
])
async def test_preserved_layout_geometry_and_inert_slots(env, app, path, mask, control):
    await env.reset()
    if '{contact}' in path:
        contact = await env.page.evaluate('() => __SIM__.getState().apps.wechat.contacts.find(c => !c.isBlacklisted).wxid')
        path = path.format(contact=contact)
    await open_route(env, app, path)
    if app == 'calendar':
        await env.page.get_by_role('button', name='更多', exact=True).click()
    target = env.page.locator(control).filter(visible=True)
    await target.wait_for()
    await env.page.evaluate('() => document.fonts.ready')
    original = await target.bounding_box()
    state = await env.page.evaluate('() => __SIM__.getState().apps')
    await env.page.evaluate('(ids) => __SIM__.setDisabledFeatures(ids, {layout:"preserve"})', mask)
    await env.page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
    assert await target.bounding_box() == pytest.approx(original, abs=0.1)
    assert await env.page.evaluate('() => __SIM__.getState().apps') == state
    slots = env.page.locator('[data-feature-layout-slot]')
    count = await slots.count()
    assert count > 0
    assert await slots.locator('button, a, input, [tabindex], [data-trigger], [data-action]').count() == 0
    for slot in await slots.all():
        assert await slot.get_attribute('aria-hidden') == 'true'
        assert await slot.evaluate('(el) => el.inert && getComputedStyle(el).visibility === "hidden"')
        assert not await slot.evaluate('(el) => { el.focus(); return document.activeElement === el; }')
        box = await slot.bounding_box()
        x, y = box['x'] + box['width']/2, box['y'] + box['height']/2
        if 0 < x < 360 and 0 < y < 800:
            await env.page.mouse.click(x, y)
            assert await slots.count() == count
            assert await env.get_route() == {'app': app, 'path': path}
            assert await env.page.evaluate('() => __SIM__.getState().apps') == state
    await env.page.evaluate('() => __SIM__.setDisabledFeatures([])')
    await env.page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
    assert await target.bounding_box() == pytest.approx(original, abs=0.1)
    assert await env.page.evaluate('() => __SIM__.getFeatureLayout().mode') == 'compact'


async def test_layout_respects_native_hidden_entries_and_page_isolation(env, request):
    await env.reset()
    await env.set_state({'apps': {'wechat': {'settings': {'discover': {'moments': {'visible': False}}}}}}, deep=True, reload=False)
    await env.page.evaluate('() => __SIM__.setDisabledFeatures(["wechat.moments"], {layout:"preserve"})')
    await open_route(env, 'wechat', '/discover')
    assert await env.page.locator('[data-feature-layout-slot]').count() == 0
    async with EnvPool(request.config.getoption('--sim-url'), n=2, num_browsers=1,
                       delay_after_action=0.1, verbose=False) as pool:
        left, right = pool[0].page, pool[1].page
        await left.evaluate('() => __SIM__.setDisabledFeatures(["notes.todo"], {layout:"preserve"})')
        assert await right.evaluate('() => __SIM__.getFeatureLayout().mode') == 'compact'
        await left.evaluate('() => __SIM__.resetState()')
        assert await left.evaluate('() => __SIM__.getFeatureLayout().mode') == 'compact'
        await left.evaluate('() => __SIM__.setDisabledFeatures(["notes.todo"], {layout:"preserve"})')
        await left.reload()
        await pool[0].wait_ready()
        assert await left.evaluate('() => __SIM__.getFeatureLayout().mode') == 'compact'
