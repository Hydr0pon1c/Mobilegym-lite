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

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const source = readFileSync('vite.config.ts', 'utf8');

describe('Vite watcher companion-data isolation', () => {
  it('does not follow or enter the large runs/data symlinks', () => {
    expect(source).toContain('followSymlinks: false');
    expect(source).toContain("'**/runs'");
    expect(source).toContain("'**/mobilegym-data'");
  });

  it('uses an explicit dependency set instead of crawling the full app graph', () => {
    expect(source).toContain('noDiscovery: true');
    expect(source).toContain("'react-dom/client'");
    expect(source).toContain("'zustand/middleware/immer'");
  });
});
