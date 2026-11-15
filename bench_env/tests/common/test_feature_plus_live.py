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

"""Browser acceptance for commerce gates and retained navigation in feature0_plus_v1."""
import pytest
from bench_env.env.pool import EnvPool

pytestmark = [pytest.mark.live, pytest.mark.asyncio(loop_scope='session')]


async def open_route(env, app, path, expected=None):
    await env.page.evaluate('([app,path]) => __OS__.openApp(app,path)', [app,path])
    await env.page.wait_for_function('([app,path]) => {const r=__OS__.getAppRoute();return r?.app===app && r.path===path}',
                                    arg=[app, expected or path])


@pytest.mark.parametrize('module,app,path,home', [
    ('bilibili.vip','bilibili','/vip','/?tab=recommend'),
    ('bilibili.shop','bilibili','/shop','/?tab=recommend'),
    ('spotify.premium','spotify','/premium','/?tab=all'),
    ('spotify.premium','spotify','/premium/payment?planId=individual','/?tab=all'),
    ('alipay.video','alipay','/video','/'),
])
async def test_plus_deep_links_block_and_reset_restores(env,module,app,path,home):
    await env.reset()
    await env.page.evaluate('(ids)=>__SIM__.setDisabledFeatures(ids)',[module])
    await open_route(env,app,path,home)
    await env.page.evaluate('()=>__SIM__.setDisabledFeatures([])')
    await open_route(env,app,path)
    await env.reset()
    assert await env.page.evaluate('()=>__SIM__.getFeatureAvailability().disabled') == []


async def test_plus_entries_and_background_pages_are_removed(env):
    await env.reset()
    await env.page.evaluate("()=>__SIM__.setDisabledFeatures(['bilibili.vip','bilibili.shop','spotify.premium','alipay.video'])")
    await open_route(env,'bilibili','/me')
    assert await env.page.locator('[data-trigger="vip.open"]').count()==0
    assert await env.page.locator('[data-trigger="tab.shop"]').count()==0
    assert await env.page.locator('[data-trigger="favorites.open"]:visible').count()==1
    assert await env.page.locator('[data-trigger="settings.open"]:visible').count()==1
    await open_route(env,'spotify','/?tab=all')
    assert await env.page.locator('[data-trigger="tab.premium"]').count()==0
    assert await env.page.locator('[data-trigger="premium.payment.open"]').count()==0
    for trigger in ('tab.search','tab.library','create.open'):
        assert await env.page.locator(f'[data-trigger="{trigger}"]:visible').count()>=1
    await open_route(env,'alipay','/')
    assert await env.page.locator('[data-trigger="tab.video"]').count()==0
    for trigger in ('tab.finance','tab.messages','tab.my'):
        assert await env.page.locator(f'[data-trigger="{trigger}"]:visible').count()==1
    await env.page.evaluate('()=>__SIM__.setDisabledFeatures([])')
    assert await env.page.locator('[data-trigger="tab.video"]:visible').count()==1


async def test_plus_masks_are_isolated_between_pages_and_reload(request):
    async with EnvPool(request.config.getoption('--sim-url'),n=2,num_browsers=1,delay_after_action=.1,verbose=False) as pool:
        left,right=pool[0].page,pool[1].page
        await left.evaluate("()=>__SIM__.setDisabledFeatures(['bilibili.shop','spotify.premium','alipay.video'])")
        assert await right.evaluate('()=>__SIM__.getFeatureAvailability().disabled')==[]
        await left.reload();await pool[0].wait_ready()
        assert await left.evaluate('()=>__SIM__.getFeatureAvailability().disabled')==[]
