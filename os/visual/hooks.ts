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

import { useMemo, type CSSProperties } from 'react';
import type { AppThemeColors } from '../types/manifest';
import { applyWechatVisual } from '../../apps/Wechat/res/visual';
import { useVisualConfig } from './service';
import { resolveVisualConfig } from './engine';
import { getVisualAdapter } from './registry';
import { applyTheme } from './theme';

export function useVisualThemeEnabled(appId: string): boolean {
  const config = useVisualConfig();
  return config.theme.strength > 0 && (config.version === 1 ? appId === 'wechat' : config.apps.includes(appId));
}

export function useAppVisualTheme(appId: string, theme: AppThemeColors, colors: Record<string, string>) {
  const config = useVisualConfig();
  const resolved = useMemo(() => resolveVisualConfig(config), [config]);
  if (config.version === 1) return appId === 'wechat' ? applyWechatVisual(theme, colors, config) : { theme, colors };
  if (!config.apps.includes(appId)) return { theme, colors };
  return applyTheme(getVisualAdapter(appId), theme, colors, resolved.theme);
}

/** Adds presentation only; callers retain their original nodes and event bindings. */
export function useVisualOrder(appId: string): (id: string) => CSSProperties | undefined {
  const config = useVisualConfig();
  const resolved = useMemo(() => resolveVisualConfig(config), [config]);
  let order: string[] | undefined;
  if ('layout' in resolved) {
    if (appId === 'wechat' && config.layout.strength !== 'native') order = resolved.layout.order;
  } else if (resolved.apps[appId]?.layoutEnabled) order = resolved.apps[appId].layout?.order;
  return id => {
    if (!order) return undefined;
    const index = order.indexOf(id);
    if (index < 0) throw new Error(`Unknown visual item: ${appId}.${id}`);
    return { order: index };
  };
}
