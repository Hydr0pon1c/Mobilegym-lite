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

/** Opt-in benchmark clock; resolved before TimeService and App stores initialize. */
export function benchmarkTime(search: string): number | undefined {
  const values = new URLSearchParams(search).getAll('benchmark_time_ms');
  if (!values.length) return undefined;
  if (values.length !== 1 || !/^[0-9]+$/.test(values[0])) {
    throw new Error('benchmark_time_ms requires one integer epoch timestamp');
  }
  const value = Number(values[0]);
  // Bound accidental seconds/NaN/overflow; this protocol uses dates in 2000–2100.
  if (!Number.isSafeInteger(value) || value < 946684800000 || value >= 4102444800000) {
    throw new Error('benchmark_time_ms must be epoch milliseconds in [2000, 2100)');
  }
  return value;
}
