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

import { describe, expect, it } from 'vitest';
import {
  collectImageSrcs,
  normalizeWmrImageSrc,
  timeDigitSrcs,
} from '../os/wmr/engine/imageCache';

describe('WMR image source collection', () => {
  it('leaves indexed-image layout detection to the renderer', () => {
    const sources = collectImageSrcs([
      { tag: 'Image', src: 'animation/frame.webp', srcid: '#frame' },
      { tag: 'Image', src: 'background.png' },
    ] as any);

    expect(sources).toEqual(['background.png']);
  });

  it('rejects transient extension-only image paths', () => {
    expect(normalizeWmrImageSrc('assets/alert/.png')).toBe('');
    expect(normalizeWmrImageSrc(' assets/alert/rain_alert.png ')).toBe('assets/alert/rain_alert.png');
    expect(normalizeWmrImageSrc('assets/alert/.webp?variant=2')).toBe('');
    expect(normalizeWmrImageSrc('assets/alert/')).toBe('');
  });

  it('does not expand an invalid Time image source', () => {
    expect(timeDigitSrcs('time/.png')).toEqual([]);
  });

  it('omits invalid static image paths from preloading', () => {
    const sources = collectImageSrcs([
      { tag: 'Image', src: 'assets/alert/.png' },
      { tag: 'ImageNumber', src: '.webp' },
      { tag: 'Time', src: 'time/.png' },
      { tag: 'Image', src: 'background.png' },
    ] as any);

    expect(sources).toEqual(['background.png']);
  });
});
