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

// Private preprocessing diagnostics. Never included in policy observations.
({ point = null, gesture = 'CLICK', probes = true } = {}) => {
  const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
  const box = el => {
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  };
  const visible = el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0'
      && r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0
      && r.left < innerWidth && r.top < innerHeight;
  };
  const selector = '[data-trigger],[data-action],input,textarea,[contenteditable="true"],button,[role="button"],[role="switch"],[role="slider"]';
  // Read-only React listener inspection locates untagged divs and nested handlers.
  // It does not replace listeners or claim that every bubbling listener fired.
  const props = el => el?.[Object.keys(el).find(k => k.startsWith('__reactProps$'))] || {};
  const eventNames = gesture === 'LONG_PRESS' ? ['onPointerDown', 'onTouchStart', 'onContextMenu']
    : gesture === 'DRAG' || gesture === 'SWIPE' ? ['onPointerDown', 'onTouchStart'] : ['onClick'];
  const handles = el => eventNames.some(name => typeof props(el)[name] === 'function');
  const elements = [...document.querySelectorAll('*')].filter(el =>
    (el.matches(selector) || handles(el)) && visible(el));
  const context = el => {
    let parent = el.parentElement;
    const own = clean(el.innerText);
    while (parent && parent !== document.body) {
      const text = clean(parent.innerText);
      if (text && text !== own && text.length <= 160) return text;
      parent = parent.parentElement;
    }
    return null;
  };
  const shape = el => [...el.querySelectorAll('svg path,svg polyline,svg polygon,svg line,circle')]
    .map(n => [n.tagName, ...['d', 'points', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r'].map(a => n.getAttribute(a))]);
  const baseIdentity = el => {
    for (const kind of ['action', 'trigger']) {
      const id = el.getAttribute(`data-${kind}`);
      if (id) {
        let params;
        try { params = JSON.parse(el.getAttribute(`data-${kind}-params`) || '{}'); }
        catch { return null; }
        return { kind, id, params, gesture: el.getAttribute(`data-${kind}-type`) };
      }
    }
    if (el.matches('input,textarea,[contenteditable="true"]')) {
      const name = el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('name');
      return name ? { kind: 'field', name, tag: el.tagName, type: el.getAttribute('type') } : null;
    }
    const label = clean(el.getAttribute('aria-label') || el.innerText);
    if (label) return { kind: 'label', label, tag: el.tagName, role: el.getAttribute('role') };
    const surrounding = context(el), icon = shape(el);
    return surrounding || icon.length ? { kind: 'context', context: surrounding,
      icon, tag: el.tagName, role: el.getAttribute('role') } : null;
  };
  const key = item => JSON.stringify(item);
  const baseCounts = new Map();
  for (const el of elements) {
    const id = baseIdentity(el);
    if (id) baseCounts.set(key(id), (baseCounts.get(key(id)) || 0) + 1);
  }
  const identity = el => {
    if (!el) return null;
    const id = baseIdentity(el);
    if (id && baseCounts.get(key(id)) > 1 && !Object.keys(id.params || {}).length) {
      const text = context(el);
      if (text) return { ...id, context: text };
    }
    return id;
  };
  const counts = new Map();
  for (const el of elements) {
    const id = identity(el);
    if (id) counts.set(key(id), (counts.get(key(id)) || 0) + 1);
  }
  const describe = el => {
    if (!el) return null;
    const id = identity(el);
    // Multiple labelled entry points with the same explicit entity parameters
    // are an intentional motor equivalence (e.g. avatar and name of one user).
    return { identity: id, unique: !!id && (counts.get(key(id)) === 1 || Object.keys(id.params || {}).length > 0),
      ownership: handles(el) ? 'react_listener' : el.matches('input,textarea,button') ? 'native_control' : 'tag_only',
      listeners: eventNames.filter(name => typeof props(el)[name] === 'function'),
      tag: el.tagName, role: el.getAttribute('role'), rect: box(el),
      label: clean(el.innerText || el.getAttribute('placeholder')).slice(0, 180),
      editable: el.matches('input,textarea,[contenteditable="true"]'),
      disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
      value: 'value' in el ? el.value : null };
  };
  const hit = (x, y) => {
    const leaf = document.elementFromPoint(x, y);
    let el = leaf;
    while (el && !handles(el) && !(gesture !== 'LONG_PRESS' && gesture !== 'DRAG' && gesture !== 'SWIPE'
      && el.matches('input,textarea,button,[contenteditable="true"]'))) el = el.parentElement;
    el ||= leaf?.closest(selector);
    const target = describe(el);
    if (target) {
      target.ancestors = [];
      let parent = el.parentElement?.closest(selector);
      while (parent) {
        target.ancestors.push(identity(parent));
        parent = parent.parentElement?.closest(selector);
      }
    }
    return target;
  };
  const target = point ? hit(...point) : (gesture === 'TYPE' || gesture === 'ENTER') ? describe(document.activeElement) : null;
  const same = (x, y) => {
    const other = hit(x, y);
    return !!target?.unique && !!other?.unique && !other.disabled
      && key(target.identity) === key(other.identity);
  };
  const grid = [], radius = [];
  if (probes && target?.unique && point) {
    const r = target.rect;
    for (const u of [0.1, 0.3, 0.5, 0.7, 0.9]) for (const v of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      const x = r.x + u * r.width, y = r.y + v * r.height;
      if (x >= 0 && x < innerWidth && y >= 0 && y < innerHeight)
        grid.push({ point: [x, y], same: same(x, y) });
    }
    // A fixed radius of 20 in normalized coordinates, sampled on its perimeter.
    for (let i = 0; i < 16; i++) {
      const angle = i * Math.PI / 8;
      const x = point[0] + 0.02 * innerWidth * Math.cos(angle);
      const y = point[1] + 0.02 * innerHeight * Math.sin(angle);
      if (x >= 0 && x < innerWidth && y >= 0 && y < innerHeight)
        radius.push({ point: [x, y], same: same(x, y) });
    }
  }
  const focus = document.activeElement;
  const domPath = el => {
    const parts = [];
    while (el && el !== document.body) {
      const siblings = [...(el.parentElement?.children || [])].filter(n => n.tagName === el.tagName);
      parts.unshift(`${el.tagName}:${siblings.indexOf(el)}`);
      el = el.parentElement;
    }
    return parts.join('/');
  };
  const scrolls = [...document.querySelectorAll('[data-scroll-container],div,main,section,ul')]
    .filter(el => visible(el) && (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1))
    .map(el => ({ id: el.getAttribute('data-scroll-container') || domPath(el), top: Math.round(el.scrollTop),
      left: Math.round(el.scrollLeft), height: el.scrollHeight, width: el.scrollWidth }));
  return {
    target, grid, radius, viewport: [innerWidth, innerHeight],
    controls: elements.map(describe),
    ui: { text: clean(document.body.innerText), scrolls,
      focus: focus?.tagName === 'BODY' ? null : {
        target: identity(focus), tag: focus?.tagName,
        value: focus?.value ?? null, start: focus?.selectionStart ?? null,
        end: focus?.selectionEnd ?? null,
      },
    },
    clock: { config: window.__SIM_TIME__?.getConfig(), now: window.__SIM_TIME__?.now(),
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone },
  };
}
