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

import externalAssetOverrides from '../data/externalAssetOverrides.json';

/**
 * 统一的 CDN 资源地址解析。
 *
 * 三端约定：
 *   dev (npm run dev) / 本地 nginx / fl1 nginx → fallback `/cdn` → 由 vite 中间件或
 *      nginx alias 映射到本地仓库根的 `mobilegym-data/`
 *   生产 (GitHub Action build) → 通过 VITE_CDN_BASE 注入完整 URL，例如
 *      `https://cdn.mobilegym.dev`，运行时直接打 R2/CF CDN
 *
 * 物理目录与 R2 完全镜像：
 *   mobilegym-data/<app>/images/...   ↔   r2:mobilegym-data/<app>/images/...
 *
 * 用法：
 *   const REDBOOK_CDN = cdn('redbook/images');
 *   const url = `${REDBOOK_CDN}/avatars/foo.jpg`;
 */

export function resolveCdnBase(raw: string | undefined): string {
  return (raw?.trim() || '/cdn').replace(/\/+$/, '');
}

const CDN_BASE = resolveCdnBase(
  (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_CDN_BASE,
);

export function cdn(subpath: string = ''): string {
  const cleaned = subpath.replace(/^\/+/, '').replace(/\/+$/, '');
  return cleaned ? `${CDN_BASE}/${cleaned}` : CDN_BASE;
}

export const CDN_ROOT = CDN_BASE;

const EXTERNAL_ASSET_OVERRIDES = externalAssetOverrides as Record<string, string | null>;

/** Hosts whose confirmed-missing URLs can be represented by an image fallback. */
const EXTERNAL_IMAGE_HOSTS = new Set([
  'api.dicebear.com',
  'external-preview.redd.it',
  'i.redd.it',
  'images.unsplash.com',
  'pbs.twimg.com',
]);
const EXTERNAL_VIDEO_HOSTS = new Set(['video.twimg.com']);

export const EXTERNAL_IMAGE_FALLBACK_URL = '/external-media-placeholder.svg';

function confirmedUnavailableFallback(raw: string): string {
  try {
    const host = new URL(raw).hostname.toLowerCase();
    if (EXTERNAL_IMAGE_HOSTS.has(host)) return EXTERNAL_IMAGE_FALLBACK_URL;
    // There is no useful deterministic substitute for video. Returning an
    // empty optional source lets the app omit the player without probing the
    // public internet or trying to decode an image placeholder as video.
    if (EXTERNAL_VIDEO_HOSTS.has(host)) return '';
  } catch {
    // A negative override for a malformed URL is still omitted below.
  }
  return '';
}

/**
 * 把 JSON 数据 / 持久化布局里的 raw 路径解析成最终 CDN URL。所有 app 公用。
 *
 * 行为表（按检查顺序）：
 *   已下载的 `http(s)://...`                → 对应本地 CDN URL
 *   已确认 404/410 的图片 URL               → 本地占位图（视频为空串）
 *   尚未确认状态的 `http(s)://...`           → 原样返回（保留“未知/瞬态失败”和“缺失”的区别）
 *   `blob:...`、`data:...`                  → 原样返回
 *   `/cdn/foo`                              → cdn('foo')   （legacy 迁移期遗留前缀）
 *   `./foo`                                 → cdn('foo')   （相对路径，跟 RedBook 数据 ./images/x 一致）
 *   `/foo`（其它绝对路径）                    → 原样返回（同源静态资源 / blob URL 兜底）
 *   `foo/bar`（裸 relative）                 → cdn('foo/bar')
 *
 * 可选 `prefix` 参数：app 数据里习惯不带自己 app 名（如 RedBook 的 ./images/avatars/x.jpg），
 * 可以传 prefix='redbook' → 最终 cdn('redbook/images/avatars/x.jpg')。
 * Launcher 数据已经在路径里带 themes/、wallpapers/ 顶级 namespace，不传 prefix 即可。
 */
export function resolveCdnUrl(raw: string, prefix?: string): string {
  if (!raw) return raw;
  if (/^https?:\/\//.test(raw)) {
    if (Object.prototype.hasOwnProperty.call(EXTERNAL_ASSET_OVERRIDES, raw)) {
      const localPath = EXTERNAL_ASSET_OVERRIDES[raw];
      if (localPath) return cdn(localPath);
      // null records a confirmed upstream 404/410. Only this verified state
      // may render a missing-resource placeholder; transient/unseen URLs stay
      // remote so they remain distinguishable and retriable.
      return confirmedUnavailableFallback(raw);
    }
    return raw;
  }
  if (/^(blob:|data:)/.test(raw)) return raw;

  let path = raw;
  if (path.startsWith('/cdn/')) {
    path = path.slice('/cdn/'.length);
  } else if (path.startsWith('./')) {
    path = path.slice(2);
  } else if (path.startsWith('/')) {
    return raw;
  }

  return cdn(prefix ? `${prefix}/${path.replace(/^\/+/, '')}` : path);
}

/** Resolve known external assets recursively in JSON fixture data. */
export function localizeKnownExternalAssets<T>(value: T): T {
  if (typeof value === 'string') {
    return (/^https?:\/\//.test(value) ? resolveCdnUrl(value) : value) as T;
  }
  if (Array.isArray(value)) {
    return value.map(item => localizeKnownExternalAssets(item)) as T;
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, localizeKnownExternalAssets(item)]),
    ) as T;
  }
  return value;
}
