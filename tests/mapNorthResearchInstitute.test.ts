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
import { extractQueriesFromTasks } from '../apps/Map/scripts/fetch_geo_snapshot.mjs';

function createLocalStorageMock() {
  const store = new Map<string, string>();
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => store.set(key, value),
    removeItem: (key: string) => store.delete(key),
    clear: () => store.clear(),
  };
}

function initialBearingDegrees(
  origin: { lat: number; lng: number },
  destination: { lat: number; lng: number },
): number {
  const toRadians = (degrees: number) => degrees * Math.PI / 180;
  const originLat = toRadians(origin.lat);
  const destinationLat = toRadians(destination.lat);
  const longitudeDelta = toRadians(destination.lng - origin.lng);
  const y = Math.sin(longitudeDelta) * Math.cos(destinationLat);
  const x =
    Math.cos(originLat) * Math.sin(destinationLat) -
    Math.sin(originLat) * Math.cos(destinationLat) * Math.cos(longitudeDelta);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

describe('NorthResearchInstituteAnswer offline data', () => {
  beforeEach(() => {
    vi.resetModules();
    Object.defineProperty(globalThis, 'localStorage', {
      value: createLocalStorageMock(),
      configurable: true,
    });
  });

  it('keeps the task-declared query in full snapshot regeneration inputs', () => {
    const queries = extractQueriesFromTasks();
    expect(queries.placeQueries).toContain('研究所');
    expect(queries.autocompleteInputs).toContain('研究所');
  });

  it('returns complete offline research institute results', async () => {
    const { searchPlacesOffline } = await import('../apps/Map/utils/offlinePlaceStore');
    const origin = placesSnapshot.location;
    const lookup = await searchPlacesOffline('研究所', origin, origin, 'zh-Hans');

    expect(lookup.results).toHaveLength(5);
    expect(lookup.results.map((item) => item.name)).toContain('中国科学院物理研究所');
    expect(lookup.results.every((item) => Number.isFinite(item.lat) && Number.isFinite(item.lng))).toBe(true);
  });

  it('makes the expected institute the closest candidate in the northeast sector', () => {
    const candidates = placesSnapshot.search_index['研究所'].map((placeId) => {
      const row = placesSnapshot.places[placeId];
      const bearing = initialBearingDegrees(placesSnapshot.location, row);
      return {
        name: typeof row.name === 'string' ? row.name : row.name.zh,
        distanceMeters: row.distanceMeters,
        northeastDeviation: Math.abs(bearing - 45),
      };
    }).filter((item) => item.northeastDeviation <= 45)
      .sort((a, b) => a.distanceMeters - b.distanceMeters);

    expect(candidates[0].name).toBe('中国科学院自动化研究所');
    expect(candidates[0].northeastDeviation).toBeLessThan(5);
    expect(candidates[1].distanceMeters).toBeGreaterThan(1_500);
  });
});
