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

import { afterEach, describe, expect, it } from 'vitest';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { Feature, FeatureRoutes, getFeatureAvailability, getFeatureLayout, isFeatureRouteAllowed, setDisabledFeatures } from '../os/featureAvailability';

afterEach(() => setDisabledFeatures([]));

describe('episode feature availability', () => {
  it.each([
    ['bilibili.vip', 'bilibili', ['/vip', '/vip?plan=year'], ['/me', '/video/1', '/search']],
    ['bilibili.shop', 'bilibili', ['/shop', '/shop/cart'], ['/me', '/favorites', '/space']],
    ['spotify.premium', 'spotify', ['/premium', '/premium/payment?planId=1'], ['/create', '/player', '/library', '/settings/privacy']],
    ['alipay.video', 'alipay', ['/video', '/video/item'], ['/finance', '/messages', '/my', '/bill']],
  ] as const)('blocks %s without cutting task helpers', (module, app, blocked, retained) => {
    setDisabledFeatures([module]);
    for (const path of blocked) expect(isFeatureRouteAllowed(app, path), path).toBe(false);
    for (const path of retained) expect(isFeatureRouteAllowed(app, path), path).toBe(true);
    expect(isFeatureRouteAllowed(app, blocked[0] + '-other')).toBe(true);
    expect(isFeatureRouteAllowed('other', blocked[0])).toBe(true);
    function ForbiddenPage() { throw new Error('unavailable page mounted'); }
    const guarded = React.createElement(FeatureRoutes, {
      app, fallback: '/?tab=recommend', children: React.createElement(ForbiddenPage),
    });
    expect(renderToString(React.createElement(MemoryRouter, { initialEntries: [blocked[0]] }, guarded))).toBe('');
  });

  it('defaults to full, validates atomically and can be reset', () => {
    expect(getFeatureAvailability().disabled).toEqual([]);
    setDisabledFeatures(['notes.todo']);
    expect(() => setDisabledFeatures(['clock.timer', 'typo'])).toThrow();
    expect(getFeatureAvailability().disabled).toEqual(['notes.todo']);
    setDisabledFeatures([]);
    expect(isFeatureRouteAllowed('notes', '/todo')).toBe(true);
  });

  it('blocks only owned routes, preserving shared media, QR and other apps', () => {
    setDisabledFeatures(['wechat.moments', 'wechat.profile_edit']);
    for (const path of ['/moments', '/moments/person?menu=camera', '/post-text-moment', '/me/signature', '/me/address/add']) {
      expect(isFeatureRouteAllowed('wechat', path), path).toBe(false);
    }
    for (const path of ['/', '/chat/person/media-picker', '/camera', '/edit-image', '/me/qrcode', '/settings', '/wechat-sports']) {
      expect(isFeatureRouteAllowed('wechat', path), path).toBe(true);
    }
    expect(isFeatureRouteAllowed('notes', '/moments')).toBe(true);
    expect(isFeatureRouteAllowed('wechat', '/moments-extra')).toBe(true);
  });

  it('never mounts unavailable pages or entries', () => {
    setDisabledFeatures(['notes.todo']);
    function ForbiddenPage() { throw new Error('must not mount'); }
    const guarded = React.createElement(FeatureRoutes, { app: 'notes', children: React.createElement(ForbiddenPage) });
    expect(renderToString(React.createElement(MemoryRouter, { initialEntries: ['/todo'] }, guarded))).toBe('');
    expect(renderToString(React.createElement(Feature, { id: 'notes.todo', children: React.createElement(ForbiddenPage) }))).toBe('');
  });

  it('preserves only an explicit placeholder, without mounting unavailable children', () => {
    setDisabledFeatures(['notes.todo'], { layout: 'preserve' });
    function ForbiddenPage() { throw new Error('must not mount'); }
    const props = { id: 'notes.todo' as const, children: React.createElement(ForbiddenPage) };
    expect(renderToString(React.createElement(Feature, props))).toBe('');
    expect(renderToString(React.createElement(Feature, {
      ...props, placeholder: React.createElement('span', { 'aria-hidden': true }, 'slot'),
    }))).toContain('aria-hidden="true"');
    expect(isFeatureRouteAllowed('notes', '/todo')).toBe(false);
    setDisabledFeatures(['notes.todo']);
    expect(getFeatureLayout().mode).toBe('compact');
    expect(renderToString(React.createElement(Feature, { ...props, placeholder: 'slot' }))).toBe('');
  });

  it('validates mask and layout atomically; reset clears both without changing the old API shape', () => {
    setDisabledFeatures(['notes.todo'], { layout: 'preserve' });
    expect(() => setDisabledFeatures(['clock.timer'], { layout: 'invalid' as 'compact' })).toThrow();
    expect(() => setDisabledFeatures(['unknown'], { layout: 'compact' })).toThrow();
    expect(getFeatureAvailability()).toEqual({ version: 'features_v1', disabled: ['notes.todo'] });
    expect(getFeatureLayout()).toEqual({ version: 'feature_layout_v1', mode: 'preserve' });
    setDisabledFeatures([]);
    expect(getFeatureAvailability()).toEqual({ version: 'features_v1', disabled: [] });
    expect(getFeatureLayout().mode).toBe('compact');
  });
});
