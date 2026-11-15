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

import { parseWmr } from './engine/parser';
import {
  collectImageSrcs,
  createPrefixedAssetUrlResolver,
  preloadAll,
  type AssetUrlResolver,
} from './engine/imageCache';
import {
  buildWmrResourceStrings,
  loadWmrResourceStrings,
  manifestMayUseResourceStrings,
} from './engine/resourceStrings';
import localeApi, { type Locale } from '../locale';
import type {
  WmrContentProviderBinder,
  WmrDocument,
  WmrNode,
  WmrProviderDependencies,
  VarValue,
} from './engine/types';
import { beginWmrPerf } from './WmrPerf';

export interface WmrBundleAnalysis {
  hasMarquee: boolean;
  hasAnimations: boolean;
  hasInteractiveNodes: boolean;
  needsMeasurePass: boolean;
  binders: WmrContentProviderBinder[];
  providerDependencies: WmrProviderDependencies;
  inferredFrameRate: number;
}

export interface WmrBundle {
  xml: string;
  doc: WmrDocument;
  resourceStrings: Record<string, VarValue>;
  staticImageSrcs: string[];
  analysis: WmrBundleAnalysis;
  assetUrlResolver: AssetUrlResolver;
  sourceKey: string;
}

export interface WmrInlineBundleSource {
  cacheKey: string;
  xml: string;
  resourceStrings?: Record<string, VarValue>;
  resourceStringFiles?: Record<string, string>;
  assetUrlResolver: AssetUrlResolver;
}

// Cache is keyed by `${baseSourceKey}#locale=${locale}` so resource strings
// for different locales don't clobber each other. We deliberately keep all
// entries for the process lifetime: only a tiny finite set of (bundle, locale)
// pairs ever exist (typically zh + en for the handful of bundles that ship
// with the app), so unbounded growth is bounded in practice. If we ever ship
// per-user locales or hundreds of bundles, revisit with an LRU.
const bundleCache = new Map<string, Promise<WmrBundle>>();

const WEATHER_PATTERN = /\b(weather_|hasweather|customEditLocalId|selected_city|weather_version|aqi\b)\b/;
const DEVICE_PATTERN = /\b(battery_level|battery_state|ChargeSpeed|applied_light_wallpaper|__miui_version_code)\b/;
const CLOCK_PATTERN = /\b(hasAlarmClock|AlarmDesk|hassteps|step_today|next_alarm_time|Mi_(?:step|begin_time|end_time)|clock_)\b/;
const MUSIC_PATTERN = /\bmusic_control\./;
const HOST_FLAGS_PATTERN = /\b(lang|wifi_state|data_state|__darkmode|isPreviewMode|enable_background_blur|is_bo_cn)\b/;
const MEASURE_PROP_PATTERN = /\.(?:bmp|text)_(?:width|height)\b/;

export async function loadWmrBundle(
  source: string | WmrInlineBundleSource,
  locale: Locale = localeApi.getLocale(),
): Promise<WmrBundle> {
  const baseSourceKey = typeof source === 'string' ? source : source.cacheKey;
  const sourceKey = `${baseSourceKey}#locale=${locale}`;
  let pending = bundleCache.get(sourceKey);
  if (pending) return pending;

  pending = (async () => {
    const stopTotal = beginWmrPerf('bundle.total', sourceKey);
    try {
      let xml = '';
      let resourceStrings: Record<string, VarValue> = {};
      let assetUrlResolver: AssetUrlResolver;

      if (typeof source === 'string') {
        const stopFetch = beginWmrPerf('bundle.fetch', sourceKey);
        const xmlUrl = source + 'manifest.xml';
        const resp = await fetch(xmlUrl);
        if (!resp.ok) {
          throw new Error(`请求 manifest 失败: ${resp.status} ${resp.statusText}`);
        }
        xml = await resp.text();
        stopFetch();
        assetUrlResolver = createPrefixedAssetUrlResolver(source);
        resourceStrings = manifestMayUseResourceStrings(xml)
          ? await loadWmrResourceStrings(source, locale)
          : {};
      } else {
        xml = source.xml;
        resourceStrings = source.resourceStringFiles
          ? buildWmrResourceStrings(source.resourceStringFiles, locale)
          : (source.resourceStrings ?? {});
        assetUrlResolver = source.assetUrlResolver;
      }

      const stopParse = beginWmrPerf('bundle.parse', sourceKey);
      const doc = parseWmr(xml);
      const analysis = analyzeWmrDocument(doc);
      const staticImageSrcs = collectImageSrcs(doc.root.children);
      stopParse();

      const stopAssets = beginWmrPerf('bundle.assets', sourceKey);
      await preloadAll(assetUrlResolver, staticImageSrcs);
      stopAssets();

      return {
        xml,
        doc,
        resourceStrings,
        staticImageSrcs,
        analysis,
        assetUrlResolver,
        sourceKey,
      };
    } catch (error) {
      bundleCache.delete(sourceKey);
      throw error;
    } finally {
      stopTotal();
    }
  })();

  bundleCache.set(sourceKey, pending);
  return pending;
}

function analyzeWmrDocument(doc: WmrDocument): WmrBundleAnalysis {
  const binders: WmrContentProviderBinder[] = [];
  const collectedStrings: string[] = [];
  let hasMarquee = false;
  let hasAnimations = doc.framerateControllers.length > 0 || doc.frameRate > 0;
  let hasInteractiveNodes = false;

  visitUnknown(doc.root, (value) => {
    if (typeof value === 'string') collectedStrings.push(value);
  });

  visitNodes(doc.root.children, (node) => {
    if ((node.tag === 'Text' || node.tag === 'DateTime') && node.marqueeSpeed) {
      hasMarquee = true;
    }
    if (node.tag === 'ContentProviderBinder') {
      binders.push(node);
    }
    if (node.tag === 'Button' || ('triggers' in node && Array.isArray((node as { triggers?: unknown[] }).triggers) && (node as { triggers?: unknown[] }).triggers!.length > 0)) {
      hasInteractiveNodes = true;
    }
    if (!hasAnimations && hasNodeAnimations(node)) {
      hasAnimations = true;
    }
  });

  const joined = collectedStrings.join('\n');
  const providerDependencies: WmrProviderDependencies = {
    weather: WEATHER_PATTERN.test(joined),
    device: DEVICE_PATTERN.test(joined),
    clock: CLOCK_PATTERN.test(joined),
    music: MUSIC_PATTERN.test(joined),
    hostFlags: HOST_FLAGS_PATTERN.test(joined),
  };

  return {
    hasMarquee,
    hasAnimations,
    hasInteractiveNodes,
    needsMeasurePass: hasInteractiveNodes || MEASURE_PROP_PATTERN.test(joined),
    binders,
    providerDependencies,
    inferredFrameRate: inferWmrDefaultFrameRate(doc, hasAnimations),
  };
}

export function inferWmrDefaultFrameRate(
  doc: Pick<WmrDocument, 'frameRate' | 'root'>,
  hasAnimations: boolean,
): number {
  if (doc.frameRate > 0) return doc.frameRate;

  // An explicit root frameRate="0" means the widget expects animation
  // controllers to wake rendering only while they are active. Falling back to
  // a synthetic default here keeps large canvas widgets repainting forever.
  if (doc.root.attrs.frameRate !== undefined) return 0;

  return hasAnimations ? 15 : 0;
}

function hasNodeAnimations(node: WmrNode): boolean {
  if ('animations' in node && Array.isArray((node as { animations?: unknown[] }).animations) && (node as { animations?: unknown[] }).animations!.length > 0) {
    return true;
  }
  if (node.tag === 'Var' && (node.animation || (node.animations?.length ?? 0) > 0)) {
    return true;
  }
  return false;
}

function visitNodes(nodes: WmrNode[], visit: (node: WmrNode) => void): void {
  for (const node of nodes) {
    visit(node);
    if ('children' in node && Array.isArray((node as { children?: WmrNode[] }).children)) {
      visitNodes((node as { children: WmrNode[] }).children, visit);
    }
    if ('normalChildren' in node && Array.isArray((node as { normalChildren?: WmrNode[] }).normalChildren)) {
      visitNodes((node as { normalChildren: WmrNode[] }).normalChildren, visit);
    }
    if ('pressedChildren' in node && Array.isArray((node as { pressedChildren?: WmrNode[] }).pressedChildren)) {
      visitNodes((node as { pressedChildren: WmrNode[] }).pressedChildren, visit);
    }
  }
}

function visitUnknown(value: unknown, visit: (value: unknown) => void): void {
  visit(value);
  if (Array.isArray(value)) {
    for (const item of value) visitUnknown(item, visit);
    return;
  }
  if (!value || typeof value !== 'object') return;
  for (const entry of Object.values(value)) {
    visitUnknown(entry, visit);
  }
}
