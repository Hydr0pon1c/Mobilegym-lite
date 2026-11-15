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

import { beforeEach, describe, expect, it, vi } from 'vitest';
import placesSnapshot from '../apps/Map/data/places.json';

function createLocalStorageMock() {
  const store = new Map<string, string>();
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => store.set(key, value),
    removeItem: (key: string) => store.delete(key),
    clear: () => store.clear(),
  };
}

describe('Map offline autocomplete integrity', () => {
  beforeEach(() => {
    vi.resetModules();
    Object.defineProperty(globalThis, 'localStorage', {
      value: createLocalStorageMock(),
      configurable: true,
    });
  });

  it('never exposes a place prediction without an offline place row', async () => {
    const { getAutocompleteOffline } = await import('../apps/Map/utils/offlinePlaceStore');
    const placeIds = new Set(Object.keys(placesSnapshot.places));

    for (const input of Object.keys(placesSnapshot.autocomplete_index)) {
      const lookup = await getAutocompleteOffline({ input, locale: 'zh-Hans' });
      for (const prediction of lookup.predictions) {
        if (prediction.kind === 'place') {
          expect(placeIds.has(prediction.place_id), `${input}: ${prediction.place_id}`).toBe(true);
        }
      }
    }
  });

  it('falls back to complete search-index rows when an exact autocomplete entry is unusable', async () => {
    const { getAutocompleteOffline } = await import('../apps/Map/utils/offlinePlaceStore');

    const exact = await getAutocompleteOffline({
      input: '万柳的北京华联',
      locale: 'zh-Hans',
      placesOnly: true,
    });
    expect(exact).toEqual({ predictions: [], source: 'miss' });

    const fallback = await getAutocompleteOffline({
      input: '万柳的北京华联',
      locale: 'zh-Hans',
      placesOnly: true,
      allowSearchIndexFallback: true,
    });
    expect(fallback.source).toBe('search_index_fallback');
    expect(fallback.predictions.length).toBeGreaterThan(0);
    expect(fallback.predictions.every((item) => item.kind === 'place')).toBe(true);
  });
});
