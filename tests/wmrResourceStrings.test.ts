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
import { manifestMayUseResourceStrings } from '../os/wmr/engine/resourceStrings';

describe('WMR resource string discovery', () => {
  it('skips graphical bundles whose @ variables are runtime state', () => {
    const xml = '<Widget><Image srcExp="@imgSrc"/><Var expression="@resultList"/></Widget>';
    expect(manifestMayUseResourceStrings(xml)).toBe(false);
  });

  it('recognizes dataset string naming conventions', () => {
    expect(manifestMayUseResourceStrings('<Text textExp="@weather_i18n"/>')).toBe(true);
    expect(manifestMayUseResourceStrings('<Text textExp="@str_cache"/>')).toBe(true);
    expect(manifestMayUseResourceStrings('<Var expression="@flag_display_talkback"/>')).toBe(true);
  });

  it('recognizes direct textual resource references', () => {
    expect(manifestMayUseResourceStrings('<Text text="@custom_label"/>')).toBe(true);
  });
});
