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

import { visualAdapter as wechat } from '../../apps/Wechat/res/visual';
import { visualAdapter as wechat_reading } from '../../apps/WechatReading/res/visual';
import { visualAdapter as calendar } from '../../system/Calendar/res/visual';
import { visualAdapter as notes } from '../../system/Notes/res/visual';
import { visualAdapter as weather } from '../../apps/Weather/res/visual';
import type { VisualAdapter } from './types';

/** Explicit, bounded registry. Declarations import no App runtime or assets. */
export const VISUAL_ADAPTERS: Readonly<Record<string, VisualAdapter>> = { wechat, wechat_reading, calendar, notes, weather };

export function getVisualAdapter(id: string): VisualAdapter {
  if (!Object.hasOwn(VISUAL_ADAPTERS, id)) throw new Error(`Unsupported visual App: ${id}`);
  return VISUAL_ADAPTERS[id];
}

export function visualCatalog(): VisualAdapter[] {
  return JSON.parse(JSON.stringify(Object.values(VISUAL_ADAPTERS)));
}
