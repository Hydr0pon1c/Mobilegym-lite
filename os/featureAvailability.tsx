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

/** Episode-local availability, separate from app data and user settings. */
import React, { useSyncExternalStore } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

export const FEATURE_VERSION = 'features_v1';
export const FEATURE_LAYOUT_VERSION = 'feature_layout_v1';
type FeatureLayout = 'compact' | 'preserve';
// Whole modules only. Shared helpers (e.g. WeChat chat media and QR) stay available.
export const FEATURE_ROUTES = {
  'bilibili.vip': ['vip'],
  'bilibili.shop': ['shop'],
  'spotify.premium': ['premium'],
  'alipay.video': ['video'],
  'clock.world': ['world'],
  'clock.stopwatch': ['stopwatch'],
  'clock.timer': ['timer'],
  'notes.todo': ['todo'],
  'notes.settings': ['settings'],
  'calendar.date_calculate': ['date-calculate'],
  'calendar.settings': ['settings'],
  'wechat.moments': ['moments', 'post-moment', 'post-text-moment', 'select-location', 'me/moments-album'],
  'wechat.profile_edit': ['my-profile-detail', 'me/name', 'me/gender', 'me/region', 'me/phone',
    'me/wxid', 'me/pat', 'me/signature', 'me/address', 'me/invoice', 'me/beans'],
} as const;
export type FeatureId = keyof typeof FEATURE_ROUTES;

let disabled: ReadonlySet<FeatureId> = new Set();
let featureLayout: FeatureLayout = 'compact';
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
};
const snapshot = () => disabled;
const layoutSnapshot = () => featureLayout;

export function setDisabledFeatures(ids: readonly string[], options?: { layout: FeatureLayout }) {
  if (!Array.isArray(ids) || ids.some(id => !Object.hasOwn(FEATURE_ROUTES, id))) {
    throw new Error('Unknown feature availability module');
  }
  const layout = options === undefined ? 'compact' : options?.layout;
  if (layout !== 'compact' && layout !== 'preserve') {
    throw new Error('Unknown feature layout');
  }
  disabled = new Set(ids as FeatureId[]);
  featureLayout = layout;
  listeners.forEach(listener => listener());
  return getFeatureAvailability();
}

export function getFeatureAvailability() {
  return { version: FEATURE_VERSION, disabled: [...disabled].sort() };
}

export function getFeatureLayout() {
  return { version: FEATURE_LAYOUT_VERSION, mode: featureLayout };
}

export function useFeatureLayout(): FeatureLayout {
  return useSyncExternalStore(subscribe, layoutSnapshot, layoutSnapshot);
}

export function useFeatureEnabled(id: FeatureId): boolean {
  return !useSyncExternalStore(subscribe, snapshot, snapshot).has(id);
}

export function isFeatureRouteAllowed(app: string, path: string, mask = disabled): boolean {
  const pathname = path.split(/[?#]/, 1)[0].replace(/^\/+|\/+$/g, '');
  return ![...mask].some(id => id.startsWith(`${app}.`) &&
    FEATURE_ROUTES[id].some(prefix => pathname === prefix || pathname.startsWith(`${prefix}/`)));
}

export function Feature({ id, children, placeholder }: {
  id: FeatureId; children: React.ReactNode; placeholder?: React.ReactNode;
}) {
  const enabled = useFeatureEnabled(id);
  const layout = useFeatureLayout();
  // Reviewed entrance bind sites supply inert presentation placeholders.
  // Never mount hidden functional children when the feature is unavailable.
  return enabled ? <>{children}</> : layout === 'preserve' ? <>{placeholder}</> : null;
}

/** A geometry-only slot. Children must be presentation, never pages or handlers. */
export function FeatureSlot({ id, className, children }: {
  id: FeatureId; className?: string; children?: React.ReactNode;
}) {
  return <div aria-hidden="true" inert data-feature-layout-slot={id} className={className}
    style={{ visibility: 'hidden', pointerEvents: 'none' }}>{children}</div>;
}

/** Defense for OS intents/deep links/history as well as ordinary app navigation.
 * Check before mounting children: unavailable pages cannot execute their effects.
 * All normal entrances must also be removed at their UI bind sites.
 */
export function FeatureRoutes({ app, children, fallback }: { app: string; children: React.ReactNode; fallback?: string }) {
  const mask = useSyncExternalStore(subscribe, snapshot, snapshot);
  const { pathname } = useLocation();
  return isFeatureRouteAllowed(app, pathname, mask)
    ? <>{children}</> : <Navigate to={fallback ?? (app === 'clock' ? '/alarm' : '/')} replace />;
}
