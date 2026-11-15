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

import pytest
from playwright.async_api import async_playwright

from bench_env.reference import DOM_PROBE


@pytest.mark.asyncio
async def test_hit_regions_exclude_nested_controls_and_clipped_parts():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            page = await browser.new_page(viewport={'width': 360, 'height': 800})
            await page.set_content('''
              <div style="position:absolute;left:10px;top:10px;width:200px;height:100px;overflow:hidden">
                <div data-action="row.open" data-action-type="tap"
                     onclick="window.hit='row'"
                     style="position:relative;width:300px;height:100px;background:#ddd">
                  Open row
                  <button data-action="row.delete" data-action-type="tap"
                          onclick="event.stopPropagation();window.hit='delete'"
                          style="position:absolute;left:100px;top:10px;width:70px;height:70px">Delete</button>
                </div>
              </div>''')
            probe = await page.evaluate(DOM_PROBE, {'point': [30, 70], 'gesture': 'CLICK'})
            assert probe['target']['identity']['id'] == 'row.open'
            assert any(not p['same'] for p in probe['grid'])
            for sample in probe['grid']:
                await page.evaluate('window.hit=null')
                await page.mouse.click(*sample['point'])
                assert sample['same'] == (await page.evaluate('window.hit') == 'row')
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_unparameterized_follow_buttons_use_distinct_row_context():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            page = await browser.new_page(viewport={'width': 360, 'height': 800})
            await page.set_content('''<div>Alice <button id="a" data-action="follow" data-action-type="tap">Follow</button></div>
                                     <div>Bob <button id="b" data-action="follow" data-action-type="tap">Follow</button></div>''')
            targets = []
            for name in ['a', 'b']:
                rect = await page.locator('#' + name).bounding_box()
                result = await page.evaluate(DOM_PROBE, {'point': [rect['x'] + 5, rect['y'] + 5], 'gesture': 'CLICK'})
                assert result['target']['unique']
                targets.append(result['target']['identity'])
            assert targets[0] != targets[1]
        finally:
            await browser.close()
