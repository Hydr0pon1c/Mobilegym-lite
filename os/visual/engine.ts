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

import coreProfile from '../../public/visual_profiles/core-v2.json';
import { configFromSearch, configUrl, hash, PALETTES, profileConfig, resolveConfig, validateConfig, type VisualConfig } from './config';
import { getVisualAdapter } from './registry';
import { resolveRow4 } from './layout';

export interface VisualConfigV2 {
  version: 2;
  apps: string[];
  theme: VisualConfig['theme'];
  layout: VisualConfig['layout'] & { apps: string[] };
  seed: number;
  sample: number;
}
export type RuntimeVisualConfig = VisualConfig | VisualConfigV2;

function exact(value: unknown, keys: string[], label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      Object.keys(value).length !== keys.length || keys.some(k => !Object.hasOwn(value, k))) throw new Error(`${label} requires exactly: ${keys.join(', ')}`);
  return value as Record<string, unknown>;
}

export function validateVisualConfig(value: unknown): RuntimeVisualConfig {
  if (value && typeof value === 'object' && 'version' in value && value.version === 1) return validateConfig(value);
  const c = exact(value, ['version', 'apps', 'theme', 'layout', 'seed', 'sample'], 'visual v2');
  if (c.version !== 2) throw new Error('Unsupported visual schema version');
  const layout = exact(c.layout, ['strength', 'variants', 'apps'], 'layout');
  const apps = (value: unknown, label: string): string[] => {
    if (!Array.isArray(value) || value.some(id => typeof id !== 'string') || new Set(value).size !== value.length) throw new Error(`${label} must be unique App IDs`);
    value.forEach(getVisualAdapter);
    return [...value];
  };
  const selected = apps(c.apps, 'apps'), layoutApps = apps(layout.apps, 'layout.apps');
  for (const id of layoutApps) {
    if (!selected.includes(id) || !getVisualAdapter(id).layout) throw new Error(`Layout unsupported or outside scope: ${id}`);
  }
  // Reuse v1's exact axis/range validation, not its Wechat-only scope/resolver.
  const axes = validateConfig({ version: 1, scope: 'wechat', theme: c.theme,
    layout: { strength: layout.strength, variants: layout.variants }, seed: c.seed, sample: c.sample });
  if (axes.theme.strength > 0 && !selected.length) throw new Error('Active theme needs an App scope');
  if (axes.layout.strength !== 'native' && !layoutApps.length) throw new Error('Active layout needs a layout App scope');
  const result: VisualConfigV2 = { version: 2, apps: selected, theme: axes.theme,
    layout: { ...axes.layout, apps: layoutApps }, seed: axes.seed, sample: axes.sample };
  Object.freeze(result.apps); Object.freeze(result.layout.apps); Object.freeze(result.layout);
  return Object.freeze(result);
}

export function profileConfigV2(name: string): VisualConfigV2 {
  const axes = profileConfig(name);
  return JSON.parse(JSON.stringify(validateVisualConfig({ ...coreProfile,
    theme: axes.theme, layout: { ...axes.layout, apps: coreProfile.layout.apps } })));
}

export function resolveVisualConfig(value: RuntimeVisualConfig) {
  const config = validateVisualConfig(value);
  if (config.version === 1) return resolveConfig(config);
  const paletteId = config.theme.palettes[(hash(`${config.seed}:theme`) % config.theme.palettes.length + config.sample) % config.theme.palettes.length];
  const apps = Object.fromEntries(config.apps.map(id => {
    const adapter = getVisualAdapter(id), group = adapter.layout;
    const enabled = config.layout.apps.includes(id);
    return [id, {
      adapterVersion: adapter.version,
      theme: { strength: config.theme.strength, mappedTokens: Object.keys(adapter.themeRoles), mappedComponents: Object.keys(adapter.componentRoles) },
      layout: group ? resolveRow4(group.group, group.items, enabled ? config.layout.strength : 'native', enabled ? config.layout.variants : 1, config.seed, config.sample) : null,
      layoutEnabled: enabled && config.layout.strength !== 'native',
      coverage: [...adapter.coverage], protected: [...adapter.protected],
    }];
  }));
  return { engineVersion: 'visual-v2.1', configId: hash(JSON.stringify(config)).toString(16).padStart(8, '0'), config,
    theme: { paletteId, palette: { ...PALETTES[paletteId] }, strength: config.theme.strength }, apps,
    coverage: { apps: [...config.apps], layoutApps: [...config.layout.apps], systemUi: 'reference',
      note: 'Declared capability, not measured episode exposure; unmapped elements retain native rendering.' } };
}

export function visualConfigFromSearch(search: string): RuntimeVisualConfig {
  const params = new URLSearchParams(search);
  if (!params.has('visual_config')) return configFromSearch(search);
  for (const key of params.keys()) {
    if ((key === 'visual' || key.startsWith('visual_')) && (key !== 'visual_config' || params.getAll(key).length !== 1)) throw new Error('visual_config cannot be combined with other visual parameters');
  }
  return validateVisualConfig(JSON.parse(params.get('visual_config')!));
}

export function visualConfigUrl(url: string, value: RuntimeVisualConfig): string {
  const config = validateVisualConfig(value);
  if (config.version === 1) return configUrl(url, config);
  const result = new URL(url);
  for (const key of [...result.searchParams.keys()]) if (key === 'visual' || key.startsWith('visual_')) result.searchParams.delete(key);
  result.searchParams.set('visual_config', JSON.stringify(config));
  return result.toString();
}
