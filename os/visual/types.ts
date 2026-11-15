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

export type ColorRole = 'accent' | 'accentDark' | 'background' | 'surface' | 'text' | 'muted' | 'border' | 'tint' | 'glass'
  // Explicit light operation pages inside an otherwise dark App (Weather).
  | 'lightBackground' | 'lightSurface' | 'lightAccent' | 'lightTint';

export interface VisualAdapter {
  readonly appId: string;
  readonly label: string;
  readonly version: number;
  readonly previewRoute: string;
  readonly themeRoles: Readonly<Partial<Record<keyof AppThemeColors, ColorRole>>>;
  readonly componentRoles: Readonly<Record<string, ColorRole>>;
  readonly coverage: readonly string[];
  readonly protected: readonly string[];
  readonly layout?: {
    readonly group: string;
    readonly kind: 'row4';
    readonly items: readonly string[];
  };
}

export const STANDARD_THEME_ROLES = {
  primary: 'accent', primaryDark: 'accentDark', background: 'background', surface: 'surface',
  textPrimary: 'text', textSecondary: 'muted', border: 'border',
} as const satisfies VisualAdapter['themeRoles'];
