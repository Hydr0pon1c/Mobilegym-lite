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
import { applyWechatVisual } from '../apps/Wechat/res/visual';
import { colors } from '../apps/Wechat/res/colors';
import type { AppThemeColors } from '../os/types/manifest';
import { blendHex, configFromSearch, configUrl, displacement, NAV_ITEMS, PALETTES, PROFILE_NAMES, profileConfig, referenceConfig, resolveConfig, validateConfig } from '../os/visual/config';
import { visualApi } from '../os/visual/service';

const native: AppThemeColors = {
  primary: '#07c160', primaryDark: '#06a152', background: '#ededed', surface: '#ffffff',
  textPrimary: '#000000', textSecondary: '#666666', border: '#e5e7eb', statusBarForeground: 'dark',
};

function contrast(a: string, b: string) {
  const luminance = (hex: string) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
    .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
  const x = luminance(a), y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

describe('visual experiment configuration', () => {
  it('keeps reference colors and layout exactly native', () => {
    const config = referenceConfig();
    const result = applyWechatVisual(native, colors, config);
    expect(result.theme).toBe(native);
    expect(result.colors).toBe(colors);
    expect(resolveConfig(config).layout.order).toEqual(NAV_ITEMS);
    expect(configFromSearch('')).toEqual(config);
  });

  it('supports every preset and lossless JSON/URL round trips', () => {
    for (const name of PROFILE_NAMES) {
      const config = profileConfig(name);
      const url = new URL(configUrl('http://localhost/?session=abc&visual=reference#anchor', config));
      expect(url.searchParams.get('session')).toBe('abc');
      expect(url.hash).toBe('#anchor');
      expect(resolveConfig(configFromSearch(url.search))).toEqual(resolveConfig(config));
    }
  });

  it('rejects typos, invalid ranges, unsupported coverage and ambiguous URLs', () => {
    for (const query of [
      '?visual=typo', '?visual_theme=1.1', '?visual_theme=NaN', '?visual_theme=',
      '?visual_seed=-1', '?visual_sample=2.5', '?visual_layout=typo', '?visual_variants=2',
      '?visual_palettes=blue,blue', '?visual_palettes=red', '?visual_them=1', '?visual_theme=1&visual_theme=0',
      '?visual=reference&visual_config={}',
    ]) expect(() => configFromSearch(query)).toThrow();
    expect(() => validateConfig({ ...referenceConfig(), scope: 'all' })).toThrow();
    expect(() => validateConfig({ ...referenceConfig(), extra: true })).toThrow();
  });

  it('exposes continuous theme strength with exact endpoints', () => {
    expect(blendHex('#000000', '#ffffff', 0.5)).toBe('#808080');
    const config = profileConfig('theme_fixed');
    config.theme.strength = 0.5;
    const half = applyWechatVisual(native, colors, config);
    expect(half.theme.primary).toBe(blendHex(native.primary, PALETTES.blue.accent, 0.5));
    config.theme.strength = 1;
    expect(applyWechatVisual(native, colors, config).theme.primary).toBe(PALETTES.blue.accent);
  });

  it('preserves warning/success/disabled/asset colors and readable known text pairs', () => {
    for (const palette of Object.keys(PALETTES) as (keyof typeof PALETTES)[]) {
      for (const strength of [0, 0.25, 0.5, 0.75, 1]) {
        const config = referenceConfig();
        config.theme = { palettes: [palette], strength };
        const result = applyWechatVisual(native, colors, config);
        for (const key of ['common_red', 'common_green', 'common_text_disabled', 'chat_list_alert_dot', 'misc_moments_cover_bg'] as const) {
          expect(result.colors[key]).toBe(colors[key]);
        }
        expect(contrast(result.theme.textPrimary, result.theme.background)).toBeGreaterThan(4.5);
        expect(contrast(result.theme.textSecondary, result.theme.surface!)).toBeGreaterThan(4.5);
        expect(contrast(result.colors.chat_bubble_text, result.colors.chat_bubble_me_bg)).toBeGreaterThan(4.5);
      }
    }
  });

  it('provides distinct equal-strength layout variants without changing the entry set', () => {
    for (const [strength, expected] of [['low', 0.125], ['medium', 0.25], ['high', 0.5]] as const) {
      for (const seed of [0, 41, 999]) {
        const config = profileConfig('layout_mix');
        config.layout.strength = strength;
        config.seed = seed;
        const variants = [];
        for (let sample = 0; sample < 3; sample++) {
          config.sample = sample;
          const result = resolveConfig(config);
          expect([...result.layout.order].sort()).toEqual([...NAV_ITEMS].sort());
          expect(displacement(result.layout.order)).toBe(expected);
          variants.push(result.layout.order.join(','));
        }
        expect(new Set(variants).size).toBe(3);
      }
    }
  });

  it('makes fixed variants sample-independent and changing one axis leave the other unchanged', () => {
    const fixed = profileConfig('combined');
    fixed.layout.variants = 1;
    fixed.theme.palettes = ['teal'];
    const a = resolveConfig(fixed);
    fixed.sample = 17;
    const b = resolveConfig(fixed);
    expect(b.layout.order).toEqual(a.layout.order);
    expect(b.theme).toEqual(a.theme);
    fixed.theme.strength = 0.3;
    expect(resolveConfig(fixed).layout).toEqual(b.layout);
  });

  it('cycles palette banks evenly and keeps exported values isolated from the engine', () => {
    const config = profileConfig('theme_mix');
    const selected = Array.from({ length: 3 }, (_, sample) => resolveConfig({ ...config, sample }).theme.paletteId);
    expect(new Set(selected).size).toBe(3);
    const before = resolveConfig(config);
    const mutable = resolveConfig(config);
    mutable.layout.bank[0].reverse();
    (mutable.theme.palette as { accent: string }).accent = '#000000';
    expect(resolveConfig(config)).toEqual(before);
  });

  it('locks configuration atomically and requires explicit unlocking', () => {
    visualApi.unlock();
    visualApi.setProfile('combined');
    const before = visualApi.snapshot();
    visualApi.lock();
    expect(() => visualApi.setProfile('reference')).toThrow(/locked/);
    expect(visualApi.snapshot().configId).toBe(before.configId);
    visualApi.unlock();
    expect(() => visualApi.setConfig({ ...referenceConfig(), seed: -1 })).toThrow();
    expect(visualApi.snapshot().configId).toBe(before.configId);
    visualApi.setProfile('reference');
  });
});
