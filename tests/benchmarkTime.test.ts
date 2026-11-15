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

import { describe, expect, it, vi, afterEach } from 'vitest';
import { benchmarkTime } from '../os/data/benchmarkTime';

describe('default-off benchmark bootstrap time', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('does not reinterpret native or visual-only URLs', () => {
    expect(benchmarkTime('')).toBeUndefined();
    expect(benchmarkTime('?visual=combined&uid=worker')).toBeUndefined();
  });

  it('accepts exact epoch milliseconds without touching Date or timers', () => {
    const originalDate = Date;
    expect(benchmarkTime('?uid=a&benchmark_time_ms=1788264000000')).toBe(1788264000000);
    expect(Date).toBe(originalDate);
  });

  it.each(['', 'nan', '1788264000', '-1', '1.5', '4102444800000', 'Infinity', '1e12'])('rejects %s', value => {
    expect(() => benchmarkTime(`?benchmark_time_ms=${value}`)).toThrow();
  });

  it('rejects duplicate clock arguments', () => {
    expect(() => benchmarkTime('?benchmark_time_ms=1788264000000&benchmark_time_ms=1788264000000')).toThrow();
  });

  it('configures the clock before any consumer imports TimeService', async () => {
    vi.stubGlobal('window', { location: { search: '?benchmark_time_ms=1788264000000' } });
    vi.resetModules();
    const { SIMULATOR_CONFIG } = await import('../os/data/simulatorConfig');
    expect(SIMULATOR_CONFIG.time).toEqual({ mode: 'simulated', simulatedTime: 1788264000000, flowing: false, speed: 1 });
  });

  it('preserves the original config with no flag', async () => {
    vi.stubGlobal('window', { location: { search: '' } });
    vi.resetModules();
    const { SIMULATOR_CONFIG } = await import('../os/data/simulatorConfig');
    expect(SIMULATOR_CONFIG.time).toEqual({ mode: 'real', simulatedTime: '2026-01-04 20:00:00', flowing: true, speed: 1 });
  });
});
