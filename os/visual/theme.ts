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

import type { AppThemeColors } from '../types/manifest';
import { blendHex, type PALETTES } from './config';
import type { VisualAdapter } from './types';

type Palette = (typeof PALETTES)[keyof typeof PALETTES];

/** Interpolate RGB channels while preserving the source alpha and exact reference. */
export function blendColor(source: string, target: string, strength: number): string {
  if (strength === 0) return source;
  const short = /^#([\da-f])([\da-f])([\da-f])$/i.exec(source);
  if (short) return blendHex(`#${short.slice(1).map(c => c + c).join('')}`, target, strength);
  if (/^#[\da-f]{6}$/i.test(source)) return blendHex(source, target, strength);
  const rgba = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(0?\.\d+|0|1(?:\.0+)?))?\s*\)$/.exec(source);
  if (!rgba || rgba.slice(1, 4).some(c => Number(c) > 255)) throw new Error(`Unsupported mapped color: ${source}`);
  const hex = `#${rgba.slice(1, 4).map(c => Number(c).toString(16).padStart(2, '0')).join('')}`;
  const mixed = blendHex(hex, target, strength);
  const rgb = [1, 3, 5].map(i => parseInt(mixed.slice(i, i + 2), 16)).join(', ');
  return rgba[4] === undefined ? `rgb(${rgb})` : `rgba(${rgb}, ${rgba[4]})`;
}

export function applyTheme(adapter: VisualAdapter, theme: AppThemeColors, colors: Record<string, string>, selection: { palette: Palette; strength: number }) {
  const { palette, strength } = selection;
  if (strength === 0) return { theme, colors };
  // Resource family, not the OS flag: many native Apps have no dark overrides.
  const rgb = /^#[\da-f]{6}$/i.test(theme.background)
    ? [1, 3, 5].map(i => parseInt(theme.background.slice(i, i + 2), 16)) : [255, 255, 255];
  const dark = rgb.reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0) < 128;
  const roles = {
    ...palette,
    background: dark ? '#181c24' : palette.background,
    surface: dark ? '#252b35' : blendHex('#ffffff', palette.background, 0.3),
    text: dark ? '#f1f3f7' : '#202735', muted: dark ? '#b9c4d4' : '#5b6575',
    border: dark ? '#526074' : '#cbd2dd', accent: dark ? palette.tint : palette.accent,
    tint: dark ? palette.accentDark : palette.tint,
    glass: palette.accentDark,
    lightBackground: palette.background, lightSurface: blendHex('#ffffff', palette.background, 0.3),
    lightAccent: palette.accent, lightTint: palette.tint,
  };
  const nextTheme = { ...theme };
  for (const [key, role] of Object.entries(adapter.themeRoles)) {
    const source = theme[key as keyof AppThemeColors];
    if (source !== undefined) Object.assign(nextTheme, { [key]: blendColor(source, roles[role], strength) });
  }
  const nextColors = { ...colors };
  for (const [key, role] of Object.entries(adapter.componentRoles)) {
    if (!(key in colors)) throw new Error(`Missing mapped color ${adapter.appId}.${key}`);
    nextColors[key] = blendColor(colors[key], roles[role], strength);
  }
  return { theme: nextTheme, colors: nextColors };
}
