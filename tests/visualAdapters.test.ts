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
import { colors as wechat } from '../apps/Wechat/res/colors';
import { colors as reading } from '../apps/WechatReading/res/colors';
import { colors as calendar } from '../system/Calendar/res/colors';
import { colors as notes } from '../system/Notes/res/colors';
import { colors as weather } from '../apps/Weather/res/colors';
import { profileConfigV2, resolveVisualConfig, validateVisualConfig } from '../os/visual/engine';
import { blendHex, PALETTES, profileConfig } from '../os/visual/config';
import { applyWechatVisual } from '../apps/Wechat/res/visual';
import { getVisualAdapter, visualCatalog } from '../os/visual/registry';
import { applyTheme } from '../os/visual/theme';

const resources: Record<string, Record<string, string>> = { wechat, wechat_reading: reading, calendar, notes, weather };
const light = { primary: '#4285f4', background: '#f4f4f4', surface: '#ffffff', textPrimary: '#111827', textSecondary: '#666666' };

describe('first-wave visual adapters', () => {
  it('declares a bounded five-App scope and three layout-capable Apps', () => {
    expect(visualCatalog().map(a => a.appId).sort()).toEqual(Object.keys(resources).sort());
    expect(profileConfigV2('combined').layout.apps.sort()).toEqual(['calendar', 'wechat', 'wechat_reading']);
    for (const id of ['notes', 'weather']) {
      const c = profileConfigV2('reference'); c.layout.apps.push(id);
      expect(() => validateVisualConfig(c)).toThrow(/Layout unsupported/);
    }
  });

  it('maps only existing supported colors and preserves every unmapped component', () => {
    for (const adapter of visualCatalog()) {
      const native = resources[adapter.appId];
      for (const palette of Object.values(PALETTES)) for (const strength of [0, 0.25, 0.5, 1]) {
        const result = applyTheme(adapter, light, native, { palette, strength });
        for (const key of Object.keys(native)) {
          if (!Object.hasOwn(adapter.componentRoles, key)) expect(result.colors[key]).toBe(native[key]);
        }
        if (strength === 0) { expect(result.theme).toBe(light); expect(result.colors).toBe(native); }
      }
    }
  });

  it('keeps reader presets/content and Weather semantic themes untouched', () => {
    const reader = applyTheme(getVisualAdapter('wechat_reading'), light, reading, { palette: PALETTES.purple, strength: 1 });
    for (const key of Object.keys(reading).filter(key => key.startsWith('reader'))) expect(reader.colors[key]).toBe(reading[key as keyof typeof reading]);
    const result = applyTheme(getVisualAdapter('weather'), light, weather, { palette: PALETTES.teal, strength: 1 });
    expect(result.theme).toEqual(light);
    expect(result.colors.chart_line_stroke).toBe(weather.chart_line_stroke);
    expect(result.colors.card_surface_detail).toMatch(/, 0\.28\)$/);
    expect(result.colors.card_surface_detail).not.toBe(weather.card_surface_detail);
  });

  it('keeps an existing App selection stable when scope expands or reorders', () => {
    const config = profileConfigV2('combined'); config.sample = 2;
    const full = resolveVisualConfig(config);
    const single = resolveVisualConfig({ ...config, apps: ['wechat'], layout: { ...config.layout, apps: ['wechat'] } });
    expect('apps' in full && full.apps.wechat).toEqual('apps' in single && single.apps.wechat);
    expect(full.theme).toEqual(single.theme);
    const reordered = resolveVisualConfig({ ...config, apps: [...config.apps].reverse(), layout: { ...config.layout, apps: [...config.layout.apps].reverse() } });
    expect('apps' in full && full.apps).toEqual('apps' in reordered && reordered.apps);
  });

  it('uses identical strength/bank rules for each four-entry adapter', () => {
    for (const strength of ['low', 'medium', 'high'] as const) {
      const c = profileConfigV2('layout_mix'); c.layout.strength = strength;
      for (const app of c.layout.apps) {
        const layouts = [0, 1, 2].map(sample => { const r = resolveVisualConfig({ ...c, sample }); if (!('apps' in r)) throw new Error('Expected v2'); return r.apps[app].layout!; });
        expect(new Set(layouts.map(l => l.order.join(','))).size).toBe(3);
        expect(new Set(layouts.map(l => l.normalizedDisplacement)).size).toBe(1);
      }
    }
  });

  it('versions the hardened adapters without changing the v1 map', () => {
    expect(visualCatalog().map(a => [a.appId, a.version])).toEqual([
      ['wechat', 3], ['wechat_reading', 2], ['calendar', 2], ['notes', 2], ['weather', 2],
    ]);
    expect(resolveVisualConfig(profileConfigV2('combined')).engineVersion).toBe('visual-v2.1');
    const v1 = applyWechatVisual(light, wechat, profileConfig('combined'));
    for (const key of ['chat_list_item_divider', 'contacts_item_divider', 'search_result_divider', 'settings_item_divider', 'tw-border-gray-100'] as const) {
      expect(v1.colors[key]).toBe(wechat[key]);
    }
  });

  it('keeps Weather operation pages light even when its home resource family is dark', () => {
    const dark = { ...light, background: '#0b1220', textPrimary: '#ffffff' };
    for (const palette of Object.values(PALETTES)) for (const strength of [0, 0.25, 0.5, 0.75, 1]) {
      const result = applyTheme(getVisualAdapter('weather'), dark, weather, { palette, strength });
      expect(result.theme).toEqual(dark);
      expect(result.colors.settings_page_bg).toBe(blendHex(weather.settings_page_bg, palette.background, strength));
      expect(result.colors.operation_surface).toBe(blendHex(weather.operation_surface, blendHex('#ffffff', palette.background, 0.3), strength));
      expect(result.colors.settings_toggle_on).toBe(blendHex(weather.settings_toggle_on, palette.accent, strength));
      expect(result.colors.settings_picker_accent).toBe(blendHex(weather.settings_picker_accent, palette.accent, strength));
      expect(result.colors.daily_chart_surface).toMatch(/,\s*0\.7\)$/);
      for (const key of ['card_surface_warning', 'temp_bar_track', 'chart_line_stroke', 'chart_dot_stroke', 'precip_bg_start', 'precip_bg_end'] as const) expect(result.colors[key]).toBe(weather[key]);
    }
  });

  it('rebinds Reading settings to identical native values without recoloring shared utility aliases', () => {
    const replacements = {
      'tw-bg-slate-100': 'settings_bg', 'tw-text-slate-800': 'settings_item_text',
      'tw-text-slate-400': 'settings_item_value_text', 'tw-border-slate-100': 'settings_item_divider',
      'tw-bg-blue-500': 'settings_toggle_on',
    } as const;
    const themed = applyTheme(getVisualAdapter('wechat_reading'), light, reading, { palette: PALETTES.purple, strength: 1 });
    for (const [alias, semantic] of Object.entries(replacements)) {
      expect(reading[alias as keyof typeof reading]).toBe(reading[semantic]);
      expect(themed.colors[alias]).toBe(reading[alias as keyof typeof reading]);
      expect(themed.colors[semantic]).not.toBe(reading[semantic]);
    }
    expect(themed.colors.settings_toggle_off).toBe(reading.settings_toggle_off);
    expect(themed.colors.settings_logout_text).toBe(reading.settings_logout_text);
  });

  it('protects Notes secondary/completed resources and keeps Calendar native component resources', () => {
    for (const palette of Object.values(PALETTES)) for (const strength of [0, 0.5, 1]) {
      const result = applyTheme(getVisualAdapter('notes'), light, notes, { palette, strength });
      for (const key of ['text_secondary', 'text_secondary_strong', 'hairline_border', 'tab_selected'] as const) expect(result.colors[key]).toBe(notes[key]);
      const cal = applyTheme(getVisualAdapter('calendar'), light, calendar, { palette, strength });
      expect(cal.colors).toEqual(calendar);
    }
  });
});
