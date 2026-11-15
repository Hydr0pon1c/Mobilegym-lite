/*
 * Copyright 2026 OPPO
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { describe, expect, it } from 'vitest';
import { PALETTES, PROFILE_NAMES, profileConfig, resolveConfig } from '../os/visual/config';
import { profileConfigV2, resolveVisualConfig, validateVisualConfig, visualConfigFromSearch, visualConfigUrl } from '../os/visual/engine';
import { rowDisplacement } from '../os/visual/layout';
import { getVisualAdapter, visualCatalog } from '../os/visual/registry';
import { applyTheme, blendColor } from '../os/visual/theme';
import { visualApi } from '../os/visual/service';
import { setSkin } from '../os/SkinService';

describe('multi-App visual core', () => {
  it('rejects actual legacy skin conflicts even when URL updates are disabled', () => {
    visualApi.unlock(); visualApi.setProfile('reference');
    try {
      setSkin('neutral', { updateUrl: false });
      expect(() => visualApi.setConfig(profileConfigV2('combined'))).toThrow(/skin=default/);
      expect(visualApi.getConfig().theme.strength).toBe(0);
      setSkin('default', { updateUrl: false });
      visualApi.setConfig(profileConfigV2('combined'));
      expect(() => setSkin('neutral', { updateUrl: false })).toThrow(/Legacy skins/);
    } finally { visualApi.setProfile('reference'); setSkin('default', { updateUrl: false }); }
  });
  it('preserves every v1 config, URL and resolved snapshot', () => {
    for (const name of PROFILE_NAMES) {
      const config = profileConfig(name);
      expect(resolveVisualConfig(config)).toEqual(resolveConfig(config));
      expect(visualConfigFromSearch(new URL(visualConfigUrl('http://local/?session=1#x', config)).search)).toEqual(config);
    }
  });

  it('validates and freezes v2 configs with lossless isolated exports', () => {
    for (const name of PROFILE_NAMES) {
      const config = profileConfigV2(name);
      const result = validateVisualConfig(config);
      expect(Object.isFrozen(result)).toBe(true);
      expect(Object.isFrozen(result.layout)).toBe(true);
      const url = new URL(visualConfigUrl('http://local/?session=1&visual=reference#x', config));
      expect(url.searchParams.get('session')).toBe('1');
      expect(url.hash).toBe('#x');
      expect(visualConfigFromSearch(url.search)).toEqual(config);
    }
    const config = profileConfigV2('combined');
    const before = resolveVisualConfig(config), mutable = resolveVisualConfig(config);
    if ('apps' in mutable) { mutable.apps.wechat.layout!.order.reverse(); mutable.apps.wechat.coverage.push('wrong'); }
    expect(resolveVisualConfig(config)).toEqual(before);
    const catalog = visualCatalog(); catalog[0].coverage = ['wrong'];
    expect(visualCatalog()[0].coverage).not.toEqual(['wrong']);
  });

  it('rejects unsupported scopes, duplicate IDs and ambiguous URL inputs', () => {
    const c = profileConfigV2('combined');
    for (const value of [
      { ...c, apps: ['unknown'] }, { ...c, apps: ['wechat', 'wechat'] },
      { ...c, apps: [] }, { ...c, extra: 1 }, { ...c, version: 3 },
      { ...c, layout: { ...c.layout, apps: [] } },
      { ...c, layout: { ...c.layout, apps: ['wechat', 'wechat'] } },
      { ...c, theme: { ...c.theme, strength: NaN } },
    ]) expect(() => validateVisualConfig(value)).toThrow();
    for (const suffix of ['&visual=reference', '&visual_typo=1', '&visual_config={}']) {
      expect(() => visualConfigFromSearch(`?visual_config=${encodeURIComponent(JSON.stringify(c))}${suffix}`)).toThrow();
    }
  });

  it('keeps distinct equal-strength layout banks and axes independent', () => {
    for (const strength of ['low', 'medium', 'high'] as const) {
      const c = profileConfigV2('layout_mix'); c.layout.strength = strength;
      const snapshots = [0, 1, 2].map(sample => resolveVisualConfig({ ...c, sample }));
      const layouts = snapshots.map(r => 'apps' in r ? r.apps.wechat.layout! : r.layout);
      expect(new Set(layouts.map(l => l.order.join(','))).size).toBe(3);
      for (const l of layouts) expect(rowDisplacement(getVisualAdapter('wechat').layout!.items, l.order)).toBe(l.normalizedDisplacement);
      c.theme = { strength: 0.8, palettes: ['purple'] };
      const changed = resolveVisualConfig(c);
      expect('apps' in changed && changed.apps.wechat.layout).toEqual(layouts[0]);
    }
  });

  it('supports alpha-preserving interpolation and exact reference identity', () => {
    expect(blendColor('rgba(0,0,0,0.3)', '#ffffff', 0.5)).toBe('rgba(128, 128, 128, 0.3)');
    expect(blendColor('#000', '#ffffff', 1)).toBe('#ffffff');
    expect(() => blendColor('red', '#ffffff', 1)).toThrow();
    const theme = { primary: '#07c160', background: '#ededed', textPrimary: '#000000', textSecondary: '#666666' };
    const colors = { anything: '#abcdef' };
    const result = applyTheme(getVisualAdapter('wechat'), theme, colors, { palette: PALETTES.blue, strength: 0 });
    expect(result.theme).toBe(theme); expect(result.colors).toBe(colors);
  });

  it('locks v2 configs and rejects invalid updates atomically', () => {
    visualApi.unlock(); visualApi.setConfig(profileConfigV2('combined')); visualApi.lock();
    const before = visualApi.snapshot();
    expect(() => visualApi.setProfile('reference')).toThrow(/locked/);
    visualApi.unlock();
    expect(() => visualApi.setConfig({ ...profileConfigV2('reference'), version: 99 })).toThrow();
    expect(visualApi.snapshot().configId).toBe(before.configId);
    visualApi.setProfile('reference');
  });
});
