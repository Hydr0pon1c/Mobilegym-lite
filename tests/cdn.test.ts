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

import { describe, expect, it } from 'vitest';
import externalAssetOverrides from '../os/data/externalAssetOverrides.json';
import {
  EXTERNAL_IMAGE_FALLBACK_URL,
  localizeKnownExternalAssets,
  resolveCdnBase,
  resolveCdnUrl,
} from '../os/utils/cdn';

describe('resolveCdnBase', () => {
  it('treats empty string as /cdn (Vite .env.local default)', () => {
    expect(resolveCdnBase('')).toBe('/cdn');
    expect(resolveCdnBase('   ')).toBe('/cdn');
  });

  it('preserves explicit remote base', () => {
    expect(resolveCdnBase('https://cdn.mobilegym.dev')).toBe('https://cdn.mobilegym.dev');
  });

  it('falls back for undefined', () => {
    expect(resolveCdnBase(undefined)).toBe('/cdn');
  });

  it('strips trailing slashes', () => {
    expect(resolveCdnBase('/cdn/')).toBe('/cdn');
    expect(resolveCdnBase('https://cdn.mobilegym.dev/')).toBe('https://cdn.mobilegym.dev');
  });
});

describe('external asset overrides', () => {
  it('maps a downloaded external URL to the local CDN', () => {
    const [remote, local] = Object.entries(externalAssetOverrides)
      .find((entry): entry is [string, string] => typeof entry[1] === 'string')!;
    expect(resolveCdnUrl(remote)).toBe(`/cdn/${local}`);
  });

  it('uses the local placeholder for an image confirmed unavailable upstream', () => {
    const [remote] = Object.entries(externalAssetOverrides)
      .find(([url, local]) => local === null && !url.includes('video.twimg.com'))!;
    expect(resolveCdnUrl(remote)).toBe(EXTERNAL_IMAGE_FALLBACK_URL);
  });

  it('does not misclassify an unseen controlled-host image as missing', () => {
    const remote = 'https://pbs.twimg.com/media/not-in-the-fixture-map.jpg';
    expect(resolveCdnUrl(remote)).toBe(remote);
  });

  it('does not misclassify an unseen controlled-host video as missing', () => {
    const remote = 'https://video.twimg.com/ext_tw_video/not-in-the-fixture-map.mp4';
    expect(resolveCdnUrl(remote)).toBe(remote);
  });

  it('leaves unrelated external URLs without an override unchanged', () => {
    const remote = 'https://example.com/not-localized.png';
    expect(resolveCdnUrl(remote)).toBe(remote);
  });

  it('localizes known URLs recursively in fixture data', () => {
    const [remote, local] = Object.entries(externalAssetOverrides)
      .find((entry): entry is [string, string] => typeof entry[1] === 'string')!;
    const fixture = { media: [remote], label: 'unchanged' };
    expect(localizeKnownExternalAssets(fixture)).toEqual({
      media: [`/cdn/${local}`],
      label: 'unchanged',
    });
  });
});
