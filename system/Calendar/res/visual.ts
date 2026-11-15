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

import { STANDARD_THEME_ROLES, type VisualAdapter } from '../../../os/visual/types';

export const visualAdapter = {
  appId: 'calendar', label: 'Calendar', version: 2, previewRoute: '/',
  themeRoles: STANDARD_THEME_ROLES, componentRoles: {},
  layout: { group: 'calendar.view-nav', kind: 'row4', items: ['year', 'month', 'week', 'day'] },
  coverage: ['Consumed home, event-form, search, date-calculation and settings theme tokens', 'Four intact view selector entries; deeper-page behavior audited without extending layout'],
  protected: ['Date grid positions, weekday order, holidays and event order', 'Native view callbacks and user settings', 'Hardcoded status/event colors and assets'],
} as const satisfies VisualAdapter;
