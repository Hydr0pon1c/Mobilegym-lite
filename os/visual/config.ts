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

/** Pure, versioned visual intervention rules. No task IDs, state writes or RNG globals. */
export const PALETTES = {
  blue: { accent: '#2457a7', accentDark: '#183e7b', tint: '#dce9fa', background: '#edf2f8' },
  purple: { accent: '#7542a8', accentDark: '#542d7c', tint: '#eee2fa', background: '#f3eff8' },
  teal: { accent: '#137568', accentDark: '#095448', tint: '#d7f1e9', background: '#ecf5f2' },
} as const;

export type PaletteId = keyof typeof PALETTES;
export type LayoutStrength = 'native' | 'low' | 'medium' | 'high';
export interface VisualConfig {
  version: 1;
  scope: 'wechat';
  theme: { strength: number; palettes: PaletteId[] };
  layout: { strength: LayoutStrength; variants: number };
  seed: number;
  /** Supplied by the caller once per rollout group, never inferred from arrival order. */
  sample: number;
}

export const NAV_ITEMS = ['wechat', 'contacts', 'discover', 'me'] as const;
export const NAV_GROUP = 'wechat.primary-nav';
const DISPLACEMENTS = { native: 0, low: 0.125, medium: 0.25, high: 0.5 } as const;

export function referenceConfig(): VisualConfig {
  return {
    version: 1, scope: 'wechat', theme: { strength: 0, palettes: ['blue'] },
    layout: { strength: 'native', variants: 1 }, seed: 41, sample: 0,
  };
}

export const PROFILE_NAMES = ['reference', 'theme_fixed', 'theme_mix', 'layout_fixed', 'layout_mix', 'combined'] as const;

export function profileConfig(name: string): VisualConfig {
  if (!(PROFILE_NAMES as readonly string[]).includes(name)) throw new Error(`Unknown visual profile: ${name}`);
  const config = referenceConfig();
  if (name.startsWith('theme') || name === 'combined') config.theme.strength = 1;
  if (name === 'theme_mix' || name === 'combined') config.theme.palettes = ['blue', 'purple', 'teal'];
  if (name.startsWith('layout') || name === 'combined') config.layout.strength = 'medium';
  if (name === 'layout_mix' || name === 'combined') config.layout.variants = 3;
  return config;
}

function object(value: unknown, keys: string[], label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  const record = value as Record<string, unknown>;
  if (Object.keys(record).some(key => !keys.includes(key)) || keys.some(key => !(key in record))) {
    throw new Error(`${label} requires exactly: ${keys.join(', ')}`);
  }
  return record;
}

/** Validate, copy and freeze so callers cannot mutate the active experiment accidentally. */
export function validateConfig(value: unknown): VisualConfig {
  const c = object(value, ['version', 'scope', 'theme', 'layout', 'seed', 'sample'], 'visual config');
  const theme = object(c.theme, ['strength', 'palettes'], 'theme');
  const layout = object(c.layout, ['strength', 'variants'], 'layout');
  if (c.version !== 1 || c.scope !== 'wechat') throw new Error('Visual v1 supports version=1, scope=wechat only');
  if (typeof theme.strength !== 'number' || !Number.isFinite(theme.strength) || theme.strength < 0 || theme.strength > 1) {
    throw new Error('theme.strength must be in [0, 1]');
  }
  if (!Array.isArray(theme.palettes) || theme.palettes.length === 0 ||
      theme.palettes.some(p => typeof p !== 'string' || !Object.hasOwn(PALETTES, p)) ||
      new Set(theme.palettes).size !== theme.palettes.length) {
    throw new Error('theme.palettes must be a nonempty unique subset of blue,purple,teal');
  }
  if (typeof layout.strength !== 'string' || !Object.hasOwn(DISPLACEMENTS, layout.strength)) {
    throw new Error('layout.strength must be native, low, medium or high');
  }
  if (!Number.isInteger(layout.variants) || Number(layout.variants) < 1 || Number(layout.variants) > 3) {
    throw new Error('layout.variants must be an integer in [1, 3]');
  }
  if (layout.strength === 'native' && layout.variants !== 1) throw new Error('Native layout has only one variant');
  for (const key of ['seed', 'sample'] as const) {
    if (!Number.isSafeInteger(c[key]) || Number(c[key]) < 0 || Number(c[key]) > 0xffffffff) {
      throw new Error(`${key} must be an unsigned 32-bit integer`);
    }
  }
  const result = {
    version: 1, scope: 'wechat',
    theme: { strength: theme.strength, palettes: [...theme.palettes] },
    layout: { strength: layout.strength, variants: layout.variants },
    seed: c.seed, sample: c.sample,
  } as VisualConfig;
  Object.freeze(result.theme.palettes);
  Object.freeze(result.theme);
  Object.freeze(result.layout);
  return Object.freeze(result);
}

/** Stable non-cryptographic fingerprint/selection hash, not a security primitive. */
export function hash(value: string): number {
  let result = 2166136261;
  for (let i = 0; i < value.length; i++) result = Math.imul(result ^ value.charCodeAt(i), 16777619);
  return result >>> 0;
}

function permutations(items: readonly string[]): string[][] {
  if (!items.length) return [[]];
  return items.flatMap((item, index) => permutations(items.filter((_, i) => i !== index)).map(rest => [item, ...rest]));
}

/** Mean center displacement / row width, for the four equal-width navigation slots. */
export function displacement(order: readonly string[]): number {
  if (order.length !== NAV_ITEMS.length || new Set(order).size !== NAV_ITEMS.length ||
      order.some(id => !(NAV_ITEMS as readonly string[]).includes(id))) throw new Error('Invalid navigation permutation');
  return order.reduce((sum, id, i) => sum + Math.abs(i - NAV_ITEMS.indexOf(id as typeof NAV_ITEMS[number])), 0) / 16;
}

const ALL_ORDERS = permutations(NAV_ITEMS);

export function resolveConfig(input: VisualConfig) {
  const config = validateConfig(input);
  const paletteIndex = (hash(`${config.seed}:theme`) % config.theme.palettes.length + config.sample) % config.theme.palettes.length;
  const paletteId = config.theme.palettes[paletteIndex];
  const candidates = ALL_ORDERS.filter(order => displacement(order) === DISPLACEMENTS[config.layout.strength]);
  candidates.sort((a, b) => {
    const difference = hash(`${config.seed}:${NAV_GROUP}:${a.join(',')}`) - hash(`${config.seed}:${NAV_GROUP}:${b.join(',')}`);
    return difference || a.join(',').localeCompare(b.join(','));
  });
  const bank = candidates.slice(0, config.layout.variants);
  if (bank.length !== config.layout.variants) throw new Error('Not enough distinct layouts at this strength');
  const order = bank[config.sample % bank.length];
  return {
    engineVersion: 'visual-v1',
    configId: hash(JSON.stringify(config)).toString(16).padStart(8, '0'),
    config,
    theme: { paletteId, palette: { ...PALETTES[paletteId] }, strength: config.theme.strength },
    layout: { group: NAV_GROUP, order: [...order], normalizedDisplacement: displacement(order), bank: bank.map(items => [...items]), availableVariants: candidates.length },
    coverage: {
      apps: ['wechat'],
      theme: 'Theme tokens and explicitly mapped component colors only; inline colors and assets retain their original appearance.',
      layoutGroups: [NAV_GROUP],
      systemUi: 'reference',
    },
  };
}

export function blendHex(source: string, target: string, strength: number): string {
  if (strength === 0) return source;
  const parse = (value: string) => {
    if (!/^#[\da-f]{6}$/i.test(value)) throw new Error(`Expected six-digit hex color: ${value}`);
    return [1, 3, 5].map(i => parseInt(value.slice(i, i + 2), 16));
  };
  const from = parse(source), to = parse(target);
  return `#${from.map((v, i) => Math.round(v + (to[i] - v) * strength).toString(16).padStart(2, '0')).join('')}`;
}

const URL_KEYS = ['visual', 'visual_config', 'visual_theme', 'visual_palettes', 'visual_layout', 'visual_variants', 'visual_seed', 'visual_sample'];

export function configFromSearch(search: string): VisualConfig {
  const params = new URLSearchParams(search);
  for (const key of params.keys()) {
    if ((key === 'visual' || key.startsWith('visual_')) && !URL_KEYS.includes(key)) throw new Error(`Unknown visual parameter: ${key}`);
    if (URL_KEYS.includes(key) && params.getAll(key).length !== 1) throw new Error(`Duplicate visual parameter: ${key}`);
  }
  if (params.has('visual_config')) {
    if (URL_KEYS.some(key => key !== 'visual_config' && params.has(key))) throw new Error('visual_config cannot be combined with visual overrides');
    return validateConfig(JSON.parse(params.get('visual_config')!));
  }
  const config = profileConfig(params.get('visual') ?? 'reference');
  const number = (key: string) => {
    const raw = params.get(key)!;
    if (!raw.trim()) throw new Error(`${key} cannot be empty`);
    return Number(raw);
  };
  if (params.has('visual_theme')) config.theme.strength = number('visual_theme');
  if (params.has('visual_palettes')) config.theme.palettes = params.get('visual_palettes')!.split(',') as PaletteId[];
  if (params.has('visual_layout')) config.layout.strength = params.get('visual_layout') as LayoutStrength;
  if (params.has('visual_variants')) config.layout.variants = number('visual_variants');
  if (params.has('visual_seed')) config.seed = number('visual_seed');
  if (params.has('visual_sample')) config.sample = number('visual_sample');
  return validateConfig(config);
}

export function configUrl(url: string, config: VisualConfig): string {
  const result = new URL(url);
  URL_KEYS.forEach(key => result.searchParams.delete(key));
  result.searchParams.set('visual_config', JSON.stringify(validateConfig(config)));
  return result.toString();
}
