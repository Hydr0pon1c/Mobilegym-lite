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

import { hash, type LayoutStrength } from './config';

export const ROW4_DISPLACEMENTS = { native: 0, low: 0.125, medium: 0.25, high: 0.5 } as const;
const permutations = (items: readonly string[]): string[][] => items.length
  ? items.flatMap((item, index) => permutations(items.filter((_, i) => i !== index)).map(rest => [item, ...rest]))
  : [[]];

export function rowDisplacement(native: readonly string[], order: readonly string[]): number {
  if (native.length !== 4 || new Set(native).size !== 4 || order.length !== 4 ||
      new Set(order).size !== 4 || order.some(id => !native.includes(id))) throw new Error('Expected a four-entry permutation');
  return order.reduce((sum, id, slot) => sum + Math.abs(slot - native.indexOf(id)), 0) / 16;
}

export function resolveRow4(group: string, items: readonly string[], strength: LayoutStrength, variants: number, seed: number, sample: number) {
  rowDisplacement(items, items);
  const candidates = permutations(items).filter(order => rowDisplacement(items, order) === ROW4_DISPLACEMENTS[strength]);
  candidates.sort((a, b) => hash(`${seed}:${group}:${a.join(',')}`) - hash(`${seed}:${group}:${b.join(',')}`) || a.join(',').localeCompare(b.join(',')));
  const bank = candidates.slice(0, variants);
  if (bank.length !== variants || variants < 1) throw new Error(`Insufficient layouts for ${group}`);
  const order = bank[sample % bank.length];
  return { group, kind: 'row4' as const, order, bank, normalizedDisplacement: rowDisplacement(items, order), availableVariants: candidates.length };
}
