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

/**
 * Image preloading and caching for WMR widgets.
 * Handles sprite sheets (srcid selects a frame from a vertically-stacked strip).
 */

const cache = new Map<string, HTMLImageElement>();
const loading = new Map<string, Promise<HTMLImageElement>>();
const failed = new Set<string>();

export type AssetUrlResolver = (src: string) => string;

export function createPrefixedAssetUrlResolver(basePath: string): AssetUrlResolver {
  return (src: string) => `${basePath}${src}`;
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  const cached = cache.get(url);
  if (cached) return Promise.resolve(cached);
  if (failed.has(url)) return Promise.resolve(cache.get(url) ?? new Image());

  let pending = loading.get(url);
  if (pending) return pending;

  pending = new Promise<HTMLImageElement>((resolve) => {
    const img = new Image();
    img.onload = () => { cache.set(url, img); loading.delete(url); resolve(img); };
    img.onerror = () => {
      cache.set(url, img);
      failed.add(url);
      loading.delete(url);
      resolve(img);
    };
    img.src = url;
  });
  loading.set(url, pending);
  return pending;
}

export function getImage(url: string): HTMLImageElement | null {
  return cache.get(url) ?? null;
}

export function isImageLoadFailed(url: string): boolean {
  return failed.has(url);
}

const IMAGE_EXTENSION_ONLY = /^\.(?:avif|bmp|gif|jpe?g|png|svg|webp)$/i;

/**
 * Reject transient dynamic paths which cannot name an image. Provider-backed
 * expressions can briefly resolve to values such as "assets/alert/.png"
 * before their selector variable is initialized; requesting those paths only
 * creates a guaranteed 404.
 */
export function normalizeWmrImageSrc(raw: unknown): string {
  const src = String(raw ?? '').trim();
  if (!src) return '';

  const path = src.split(/[?#]/, 1)[0].replace(/\\/g, '/');
  if (path.endsWith('/')) return '';
  const basename = path.slice(path.lastIndexOf('/') + 1);
  if (!basename || IMAGE_EXTENSION_ONLY.test(basename)) return '';
  return src;
}

/**
 * Derive per-character image URLs from a Time element's src path.
 * E.g. "time/0/t.png" → ["time/0/t_0.png", ..., "time/0/t_9.png", "time/0/t_dot.png"]
 */
export function timeDigitSrcs(srcBase: string): string[] {
  const normalized = normalizeWmrImageSrc(srcBase);
  if (!normalized) return [];
  const dot = normalized.lastIndexOf('.');
  const stem = dot >= 0 ? normalized.slice(0, dot) : normalized;
  const ext = dot >= 0 ? normalized.slice(dot) : '.png';
  const srcs: string[] = [];
  for (let d = 0; d <= 9; d++) srcs.push(`${stem}_${d}${ext}`);
  srcs.push(`${stem}_dot${ext}`);
  return srcs;
}

/**
 * Collect all image src references from an AST node tree.
 */
export function collectImageSrcs(nodes: import('./types').WmrNode[]): string[] {
  const srcs = new Set<string>();
  function walk(ns: import('./types').WmrNode[]) {
    for (const n of ns) {
      // Indexed images can be either a sprite sheet or separate `_N` files.
      // WmrRenderer probes the current variant first and falls back to the
      // direct sheet; preloading the direct path here creates a guaranteed 404
      // for split-frame bundles.
      const imageSrc = 'src' in n ? normalizeWmrImageSrc(n.src) : '';
      if (n.tag === 'Image' && imageSrc && !n.srcid) srcs.add(imageSrc);
      if (n.tag === 'ImageNumber' && imageSrc) srcs.add(imageSrc);
      if (n.tag === 'Time' && n.src) {
        for (const s of timeDigitSrcs(n.src)) srcs.add(s);
      }
      if ('children' in n && Array.isArray((n as any).children)) {
        walk((n as any).children);
      }
      if ('normalChildren' in n && Array.isArray((n as any).normalChildren)) {
        walk((n as any).normalChildren);
      }
      if ('pressedChildren' in n && Array.isArray((n as any).pressedChildren)) {
        walk((n as any).pressedChildren);
      }
    }
  }
  walk(nodes);
  return [...srcs];
}

/**
 * Preload all images from a WMR widget.
 * @param basePath  URL prefix, e.g. "/themes/<themeId>/clock_2x4/"
 * @param srcs      Relative src paths from collectImageSrcs
 */
export async function preloadAll(
  basePathOrResolver: string | AssetUrlResolver,
  srcs: string[],
): Promise<void> {
  const resolveUrl = typeof basePathOrResolver === 'string'
    ? createPrefixedAssetUrlResolver(basePathOrResolver)
    : basePathOrResolver;
  const validSrcs = srcs.map(normalizeWmrImageSrc).filter(Boolean);
  await Promise.all(validSrcs.map(s => loadImage(resolveUrl(s))));
}

/**
 * Draw a sprite frame from a vertically-stacked sprite strip.
 * WMR sprite images are stacked vertically: frame 0 at top, frame 1 below, etc.
 * Each frame has width = img.naturalWidth, height = img.naturalHeight / frameCount.
 * frameCount is inferred from aspect ratio (assumes square-ish frames).
 */
export function drawSpriteFrame(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  frameIndex: number,
  dx: number, dy: number, dw: number, dh: number,
): void {
  if (!img.naturalWidth || !img.naturalHeight) return;
  const iw = img.naturalWidth;
  const ih = img.naturalHeight;

  // Estimate frame count: height/width gives approximate count
  const frameCount = Math.max(1, Math.round(ih / iw));
  const frameH = ih / frameCount;
  const fi = Math.max(0, Math.min(Math.floor(frameIndex), frameCount - 1));

  ctx.drawImage(img, 0, fi * frameH, iw, frameH, dx, dy, dw, dh);
}
