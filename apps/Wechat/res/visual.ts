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

/** Rendering-only pilot adapter. Unknown colors and business status colors stay native. */
import type { AppThemeColors } from '../../../os/types/manifest';
import { blendHex, resolveConfig, type VisualConfig } from '../../../os/visual/config';
import { STANDARD_THEME_ROLES, type VisualAdapter } from '../../../os/visual/types';

const COMPONENT_ROLES = {
  chat_list_item_bg: 'surface', chat_list_item_bg_sticky: 'background', chat_list_item_bg_active: 'background',
  chat_list_item_name: 'text', chat_list_item_preview: 'muted', chat_list_item_time: 'muted',
  chat_bubble_me_bg: 'tint', chat_bubble_other_bg: 'surface', chat_bubble_text: 'text',
  chat_input_bar_bg: 'background', chat_input_bar_border: 'border',
  search_bar_bg: 'background', search_input_bg: 'surface', search_icon_color: 'muted',
  search_cancel_text: 'accent', search_filter_tab_text: 'text', search_filter_tab_active_indicator: 'accent',
  search_category_link_text: 'accent', search_empty_text: 'muted',
  contacts_item_bg_active: 'background', contacts_section_header_bg: 'background',
  contacts_section_header_text: 'muted', contacts_count_text: 'muted',
  me_username: 'text', me_wxid_text: 'muted', me_avatar_bg: 'background',
  settings_item_text: 'text', settings_item_extra_text: 'muted', settings_group_title_text: 'muted',
  discover_item_bg: 'surface', address_link_text: 'accent',
  common_text_primary: 'text', common_text_secondary: 'text', common_text_tertiary: 'muted', common_text_hint: 'muted',
  'tw-text-gray-400': 'muted', 'tw-text-gray-500': 'muted', 'tw-text-gray-700': 'text',
  'tw-bg-gray-50': 'background', 'tw-bg-gray-100': 'background',
} as const;

export const visualAdapter = {
  appId: 'wechat', label: 'Wechat', version: 3, previewRoute: '/me',
  // Keep COMPONENT_ROLES frozen: the v1 renderer below still uses that exact map.
  themeRoles: STANDARD_THEME_ROLES, componentRoles: {
    ...COMPONENT_ROLES,
    chat_list_item_divider: 'border', contacts_item_divider: 'border',
    search_result_divider: 'border', settings_item_divider: 'border',
    'tw-border-gray-100': 'border',
  },
  layout: { group: 'wechat.primary-nav', kind: 'row4', items: ['wechat', 'contacts', 'discover', 'me'] },
  coverage: ['Theme tokens and explicitly mapped chat, contacts, search, settings and navigation colors', 'Shared settings/privacy row dividers; profile/settings surfaces and controls'],
  protected: ['Warnings, success, unread dots, disabled colors', 'Assets and unmapped inline colors', 'Business state, user settings and content order'],
} as const satisfies VisualAdapter;

export function applyWechatVisual(theme: AppThemeColors, colors: Record<string, string>, config: VisualConfig) {
  const strength = config.theme.strength;
  if (strength === 0) return { theme, colors };
  const palette = resolveConfig(config).theme.palette;
  // Classify the incoming resource palette, not the OS flag: Wechat currently
  // has empty dark overrides. Interpolating light resources toward dark colors
  // would collapse text/background contrast halfway through the interpolation.
  const channels = [1, 3, 5].map(i => parseInt(theme.background.slice(i, i + 2), 16));
  const isDark = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2] < 128;
  const roles = {
    ...palette,
    // Stay in the same light/dark family as the incoming resource palette.
    background: isDark ? '#181c24' : palette.background,
    surface: isDark ? '#252b35' : '#ffffff',
    text: isDark ? '#f1f3f7' : '#202735',
    muted: isDark ? '#b9c4d4' : '#5b6575',
    border: isDark ? '#526074' : '#cbd2dd',
    accent: isDark ? palette.tint : palette.accent,
    tint: isDark ? palette.accentDark : palette.tint,
  };
  const mix = (source: string, target: string) => blendHex(source, target, strength);
  const nextTheme: AppThemeColors = {
    ...theme,
    primary: mix(theme.primary, roles.accent),
    primaryDark: mix(theme.primaryDark ?? theme.primary, roles.accentDark),
    onPrimary: theme.onPrimary ?? '#ffffff',
    background: mix(theme.background, roles.background),
    surface: mix(theme.surface ?? '#ffffff', roles.surface),
    textPrimary: mix(theme.textPrimary, roles.text),
    textSecondary: mix(theme.textSecondary, roles.muted),
    border: mix(theme.border ?? '#e5e7eb', roles.border),
  };
  const nextColors = { ...colors };
  for (const [key, role] of Object.entries(COMPONENT_ROLES)) {
    if (colors[key] !== undefined) nextColors[key] = mix(colors[key], roles[role]);
  }
  return { theme: nextTheme, colors: nextColors };
}
