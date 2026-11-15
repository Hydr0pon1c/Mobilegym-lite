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

import type { VisualAdapter } from '../../../os/visual/types';
import { colors } from './colors';

/** Preserve alpha and all informational colors. Only selected surfaces opt in. */
export const visualCardColors = {
  ...colors, card_surface: 'var(--app-c-card-surface)',
  card_surface_light: 'var(--app-c-card-surface-light)', card_surface_detail: 'var(--app-c-card-surface-detail)',
  chart_section_bg: 'var(--app-c-chart-section-bg)',
};

export const visualAdapter = {
  appId: 'weather', label: 'Weather', version: 2, previewRoute: '/',
  themeRoles: {},
  componentRoles: {
    card_surface: 'glass', card_surface_light: 'glass', card_surface_detail: 'glass', chart_section_bg: 'glass',
    settings_page_bg: 'lightBackground', operation_surface: 'lightSurface',
    settings_toggle_on: 'lightAccent', settings_picker_accent: 'lightAccent',
    city_search_bg: 'lightBackground', city_search_input_bg: 'lightSurface', city_search_selected_bg: 'lightTint',
    daily_page_bg: 'lightBackground', daily_chart_surface: 'lightSurface',
  },
  coverage: ['Selected home/hourly card surfaces; original alpha preserved', 'Light settings/privacy/permissions surfaces, toggle-on and unit-picker controls', 'City search surfaces and daily forecast container; chart data/colors untouched'],
  protected: ['Weather-condition background/animation, AQI and warning colors, chart/gauge colors', 'Hourly/daily/city order and all layout (unsupported)', 'Setting values, units, all content and unmapped colors; native dark CSS overrides'],
} as const satisfies VisualAdapter;
