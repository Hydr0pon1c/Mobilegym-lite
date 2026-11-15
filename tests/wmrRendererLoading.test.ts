/*
 * Copyright 2026 OPPO
 *
 * Modified by OPPO for this release.
 * OPPO copyright applies only to the modifications.
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

const source = readFileSync('os/wmr/WmrRenderer.tsx', 'utf8');

describe('WMR renderer loading state', () => {
  it('uses a generic loading label for widget loading', () => {
    expect(source).not.toContain('正在加载 WMR');
    expect(source).toContain('Loading...');
  });

  it('runs init commands before resolving dynamic image sources', () => {
    const init = source.indexOf("vars.executeTriggerList(doc.root.externalTriggers, 'init')");
    const dynamicPreload = source.indexOf("const { evalExprStr: evalE, toNum }");

    expect(init).toBeGreaterThan(-1);
    expect(dynamicPreload).toBeGreaterThan(init);
  });

  it('normalizes dynamic image sources before preloading them', () => {
    expect(source).toContain('const resolved = normalizeWmrImageSrc(evalE(n.srcExp, vars))');
    expect(source).toContain('const resolved = normalizeWmrImageSrc(rawSrc)');
  });
});
