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

import { useSyncExternalStore } from 'react';
import { PROFILE_NAMES, profileConfig, referenceConfig } from './config';
import { profileConfigV2, resolveVisualConfig, validateVisualConfig, visualConfigFromSearch, visualConfigUrl, type RuntimeVisualConfig } from './engine';
import { visualCatalog } from './registry';
import { getActiveSkinId } from '../SkinService';

let config = validateVisualConfig(referenceConfig());
let locked = false;
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };
const getConfig = () => config;

export function assertLegacySkinAllowed(id: string): void {
  if (id !== 'default' && (config.theme.strength > 0 || config.layout.strength !== 'native')) {
    throw new Error('Legacy skins cannot be combined with an active visual experiment');
  }
}

export function useVisualConfig(): RuntimeVisualConfig {
  return useSyncExternalStore(subscribe, getConfig, getConfig);
}

function rejectLegacySkin(next: RuntimeVisualConfig): void {
  if (next.theme.strength === 0 && next.layout.strength === 'native') return;
  const skin = typeof window === 'undefined' ? null : new URLSearchParams(window.location.search).get('skin');
  if (getActiveSkinId() !== 'default' || (skin && skin !== 'default')) throw new Error('Use visual profiles with skin=default; legacy image filters would confound this experiment');
}

function setConfig(value: unknown): void {
  if (locked) throw new Error('Visual configuration is locked for this episode; unlock explicitly before starting a new group');
  const next = validateVisualConfig(value);
  rejectLegacySkin(next);
  resolveVisualConfig(next);
  if (typeof window !== 'undefined') window.history.replaceState(null, '', visualConfigUrl(window.location.href, next));
  config = next;
  listeners.forEach(listener => listener());
}

export const visualApi = {
  profiles: () => [...PROFILE_NAMES],
  preset: (name: string) => profileConfig(name),
  presetV2: (name: string) => profileConfigV2(name),
  catalog: visualCatalog,
  getConfig: () => JSON.parse(JSON.stringify(config)) as RuntimeVisualConfig,
  snapshot: () => ({ ...resolveVisualConfig(config), locked }),
  setConfig,
  setProfile: (name: string) => setConfig(profileConfig(name)),
  lock: () => { locked = true; },
  unlock: () => { locked = false; },
};

declare global { interface Window { __VISUAL__: typeof visualApi; } }

export function initVisualConfig(): void {
  const next = visualConfigFromSearch(window.location.search);
  rejectLegacySkin(next);
  config = next;
  locked = false;
  window.__VISUAL__ = visualApi;
}
