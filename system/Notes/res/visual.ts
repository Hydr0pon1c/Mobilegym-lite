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
import { colors } from './colors';

/** Explicit opt-in for audited pages; other resource values remain native. */
export const visualPageColors = {
  ...colors, page_bg: 'var(--app-c-page-bg)', theme_main: 'var(--app-c-theme-main)',
};
// Retain the original export for the list consumers.
export const visualListColors = visualPageColors;

export const visualAdapter = {
  appId: 'notes', label: 'Notes', version: 2, previewRoute: '/',
  themeRoles: STANDARD_THEME_ROLES,
  componentRoles: { page_bg: 'background', theme_main: 'accent' },
  coverage: ['Consumed theme tokens', 'Notes/todo, folder, settings, private/trash backgrounds and accent controls', 'Editor folder chip and input-dialog confirmation; editor surface through theme tokens'],
  protected: ['Two-tab navigation (layout unsupported)', 'Note/todo ordering, folder structure and user view mode', 'Icons, editor warning/reminder colors, disabled/completed states and unmapped colors'],
} as const satisfies VisualAdapter;
