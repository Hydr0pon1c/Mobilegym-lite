# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental reference annotations and binary path scoring; no policy inputs."""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

DOM_PROBE = Path(__file__).with_name('reference_dom.js').read_text()


def changed_paths(before, after, prefix='') -> list[str]:
    if type(before) is not type(after):
        return [prefix]
    if isinstance(before, dict):
        paths = []
        for key in sorted(before.keys() | after.keys()):
            path = f'{prefix}.{key}' if prefix else key
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(changed_paths(before[key], after[key], path))
        return paths
    if isinstance(before, list):
        if len(before) != len(after):
            return [prefix]
        return [path for i, (a, b) in enumerate(zip(before, after))
                for path in changed_paths(a, b, f'{prefix}[{i}]')]
    return [] if before == after else [prefix]


def canonical_route(route: dict | None) -> dict:
    route = route or {}  # The desktop has no foreground app/route.
    url = urlsplit(route.get('path') or '/')
    query = urlencode(sorted(parse_qsl(url.query, keep_blank_values=True)))
    return {'app': route.get('app'), 'path': url.path + ('?' + query if query else '')}


def created_entities(before: dict, after: dict) -> list[dict]:
    """Bind newly inserted string IDs only in collections present before the action."""
    def collect(value, path=(), collections=None, records=None):
        collections = set() if collections is None else collections
        records = [] if records is None else records
        if isinstance(value, dict):
            for key, child in value.items():
                collect(child, (*path, key), collections, records)
        elif isinstance(value, list):
            collections.add(path)
            for item in value:
                if isinstance(item, dict) and isinstance(item.get('id'), str):
                    records.append((path, item['id']))
                collect(item, (*path, '[]'), collections, records)
        return collections, records
    collections, old = collect(before)
    _, new = collect(after)
    old_ids = {value for _, value in old}
    return [{'collection': list(path), 'id': value} for path, value in new
            if path in collections and value not in old_ids]


def created_provider_entities(before_state: dict, after_state: dict) -> list[dict]:
    """OS-owned data (e.g. newly sent SMS conversations) also creates route IDs."""
    return created_entities({'@providers': before_state.get('os', {}).get('providers', {})},
                            {'@providers': after_state.get('os', {}).get('providers', {})})


def target_identity(step: dict) -> dict | None:
    target = step['before']['target']
    if not target or target['disabled'] or target['identity'] is None:
        return None
    if target['unique']:
        return target['identity']
    # A repeated action tag with omitted parameters can still have a unique,
    # visible row label. This is a measured text fallback, not an entity ID.
    identity, label = target['identity'], target.get('label')
    controls = step['before'].get('controls', [])
    matches = [c for c in controls if c['identity'] == identity and c.get('label') == label]
    if label and len(matches) == 1:
        return {**identity, 'row_label': label}
    return None


def operation_token(step: dict) -> dict | None:
    """Conservative prototype. Unresolved controls/gestures never become coordinates."""
    action = step['action_type']
    token = {'action': action, 'route': canonical_route(step['route'])}
    if action in {'COMPLETE', 'ANSWER', 'BACK', 'HOME', 'ENTER'}:
        return token
    if action == 'AWAKE':
        return {**token, 'app': step['action_data'].get('value')}
    if action in {'CLICK', 'LONG_PRESS', 'DOUBLE_TAP', 'TYPE'}:
        target = step['before']['target']
        identity = target_identity(step)
        if identity is None:
            # Empty full-screen React backdrops can be identified by removing
            # only modal query state, with no app mutation and no path change.
            before_route, after_route = step['route'] or {}, step.get('next_route') or {}
            before_url = urlsplit(before_route.get('path') or '/')
            after_url = urlsplit(after_route.get('path') or '/')
            before_query, after_query = dict(parse_qsl(before_url.query)), dict(parse_qsl(after_url.query))
            removed = {k: v for k, v in before_query.items() if k not in after_query}
            viewport = step['before'].get('viewport', [0, 0])
            rect = (target or {}).get('rect', {})
            if (action == 'CLICK' and target and not target['disabled']
                and target.get('ownership') == 'react_listener' and 'onClick' in target.get('listeners', [])
                and not step.get('app_changed_paths') and removed
                and before_url.path == after_url.path and before_route.get('app') == after_route.get('app')
                and all(before_query.get(k) == v for k, v in after_query.items())
                and rect.get('width', 0) >= viewport[0] * 0.95 and rect.get('height', 0) >= viewport[1] * 0.95):
                return {**token, 'target': {'kind': 'dismiss_backdrop', 'removed_query': removed}}
            return None
        expected = {'CLICK': 'tap', 'LONG_PRESS': 'longPress', 'DOUBLE_TAP': 'doubleTap'}.get(action)
        if (identity['kind'] in {'action', 'trigger'} and expected
                and identity.get('gesture') not in {expected, 'back'}
                and target.get('ownership') != 'react_listener'):
            return None
        token['target'] = identity
        if action == 'TYPE':
            token['value'] = step['action_data'].get('value', '')
            token['clear'] = bool(step['action_data'].get('clear', False))
        return token
    if action == 'SWIPE' and not step.get('app_changed_paths') and step['route'] == step['next_route']:
        before = {s['id']: s for s in step['before']['ui']['scrolls']}
        after = {s['id']: s for s in step['after']['ui']['scrolls']}
        motion = []
        for key in sorted(before.keys() & after.keys()):
            for axis in ('left', 'top'):
                delta = after[key][axis] - before[key][axis]
                if delta:
                    motion.append({'container': key, 'axis': axis, 'direction': 1 if delta > 0 else -1})
        return {**token, 'scroll': motion} if motion else None
    if action == 'DRAG':
        identity = target_identity(step)
        if identity and step.get('app_effects'):
            return {**token, 'target': identity, 'effects': step['app_effects']}
    # Scroll/drag equivalence needs an effect witness; not inferred from direction alone.
    return None


def normalized_tokens(steps: list[dict], ignored=(), *, tokenizer=operation_token, collapse_scroll=True) -> list[dict | None]:
    aliases, counts = {}, defaultdict(int)
    tokens = []
    for step in steps:
        for entity in step.get('created_entities', []):
            key = '/'.join(entity['collection'])
            if entity['id'] not in aliases:
                counts[key] += 1
                aliases[entity['id']] = f'<new:{key}:{counts[key]}>'
        if step['source_step'] in ignored:
            continue
        token = deepcopy(tokenizer(step))
        if token is not None:
            url = urlsplit(token['route']['path'])
            path = '/'.join(quote(aliases.get(unquote(part), unquote(part)), safe='<>:') for part in url.path.split('/'))
            query = urlencode([(k, aliases.get(v, v)) for k, v in parse_qsl(url.query, keep_blank_values=True)])
            token['route']['path'] = path + ('?' + query if query else '')
            if token.get('target', {}).get('params'):
                token['target']['params'] = {k: aliases.get(v, v) if isinstance(v, str) else v
                                             for k, v in token['target']['params'].items()}
        # Consecutive, verified pure scrolls of the same container/direction
        # may have different counts. Other gestures and subgoal order stay fixed.
        if collapse_scroll and token and token.get('scroll') and tokens and _semantic(tokens[-1]) == _semantic(token):
            continue
        tokens.append(token)
    return tokens


def cleaning_candidates(steps: list[dict]) -> list[dict]:
    """Propose deletions for fresh replay validation, never apply them here."""
    candidates = []
    for index, step in enumerate(steps):
        if step['action_type'] == 'WAIT' and step.get('noop_candidate'):
            candidates.append({'source_step': step['source_step'], 'reason': 'unchanged_wait'})
        if index + 1 < len(steps):
            following = steps[index + 1]
            if (step['action_type'] == 'CLICK' and following['action_type'] == 'TYPE'
                and following['action_data'].get('point') == step['action_data'].get('point')
                and step['before']['target'] and step['before']['target'].get('editable')
                and target_identity(step) is not None and target_identity(step) == target_identity(following)):
                candidates.append({'source_step': step['source_step'], 'reason': 'type_retaps_same_field'})
    return candidates


def compare_paths(reference: list[dict], candidate: list[dict], *, ignored_reference=(), ignored_candidate=()) -> dict:
    """Match all retained operations, preserving order; unknown is not a match.

    Ignored source steps must come from separately validated cleaning. The matcher
    itself never guesses that a wait, repeated click or detour was ineffective.
    """
    left = normalized_tokens(reference, ignored_reference)
    right = normalized_tokens(candidate, ignored_candidate)
    if any(s is None for s in left + right):
        return {'status': 'unresolved', 'match': False}
    return {'status': 'match' if left == right else 'different', 'match': left == right}


def binary_rewards(native_success: bool, comparison: dict) -> dict:
    """Unresolved annotations are ineligible, rather than training negatives."""
    if type(native_success) is not bool:
        raise TypeError('native_success must be the strict boolean verdict, not progress')
    return {'native': int(native_success),
            'restricted': None if comparison['status'] == 'unresolved'
            else int(native_success and comparison['match']),
            'eligible': comparison['status'] != 'unresolved'}


def _semantic(token):
    return {k: v for k, v in token.items() if not k.startswith('_')} if token else token


def practical_ignored(steps: list[dict]) -> set[int]:
    """Opt-in approximate cleaning, applied symmetrically to both paths.

    UI text, scroll, focus, route and app state must all be unchanged before
    dropping a motor no-op. This is a heuristic, not an exhaustive state proof.
    """
    ignored = {r['source_step'] for r in cleaning_candidates(steps)}
    ignored.update(s['source_step'] for s in steps if s.get('noop_candidate')
                   and s['action_type'] in {'CLICK', 'SWIPE', 'WAIT'})
    return ignored


def practical_operation_token(step: dict) -> dict:
    """Keep semantic targets where possible; retain motor evidence for fallback.

    The fallback is only used when the *reference* lacks a semantic target.
    It is not a wildcard for unrecognized candidate actions.
    """
    token = operation_token(step)
    if token is None:
        token = {'action': step['action_type'], 'route': canonical_route(step['route']),
                 'fallback': True}
    if step['action_type'] == 'AWAKE':
        token['app'] = (step.get('next_route') or {}).get('app') or token.get('app')
    target = step.get('before', {}).get('target') or {}
    rect, viewport = target.get('rect'), step.get('before', {}).get('viewport')
    bounds = None
    if rect and viewport and 0 < rect['width'] * rect['height'] < viewport[0] * viewport[1] * 0.4:
        bounds = [1000 * rect['x'] / viewport[0], 1000 * rect['y'] / viewport[1],
                  1000 * (rect['x'] + rect['width']) / viewport[0],
                  1000 * (rect['y'] + rect['height']) / viewport[1]]
    token['_motor'] = deepcopy(step['action_data'])
    token['_bounds'] = bounds
    return token


def practical_tokens(steps: list[dict], *, clean=True) -> list[dict]:
    return normalized_tokens(steps, practical_ignored(steps) if clean else (),
                             tokenizer=practical_operation_token, collapse_scroll=clean)


def compare_practical_tokens(reference: list[dict], candidate: list[dict]) -> dict:
    """Binary ordered matching with explicitly bounded reference fallbacks.

    Unknown taps use the observed control rectangle, or a radius of 20/1000
    when no small rectangle exists. Unknown swipes preserve direction and start
    region; drags preserve both endpoints. Payload and clear semantics stay fixed.
    """
    def point_matches(point, expected, bounds=None, radius=20):
        if point is None or expected is None:
            return point == expected
        if bounds:
            return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]
        return sum((x-y)**2 for x, y in zip(point, expected)) <= radius**2

    def direction(data):
        start, end = data.get('point1'), data.get('point2')
        if start is None or end is None:
            return None
        delta = [end[i] - start[i] for i in (0, 1)]
        axis = int(abs(delta[1]) > abs(delta[0]))
        return axis, (delta[axis] > 0) - (delta[axis] < 0)

    def equivalent(left, right):
        if not left.get('fallback'):
            return _semantic(left) == _semantic(right)
        if left['action'] != right['action'] or left['route'] != right['route']:
            return False
        a, b = left['_motor'], right['_motor']
        action = left['action']
        if action in {'CLICK', 'LONG_PRESS', 'DOUBLE_TAP', 'TYPE'}:
            if action == 'TYPE' and (a.get('value', '') != b.get('value', '')
                                    or bool(a.get('clear')) != bool(b.get('clear'))):
                return False
            return point_matches(b.get('point'), a.get('point'), left.get('_bounds'))
        if action == 'SWIPE':
            return (direction(a) is not None and direction(a) == direction(b)
                    and point_matches(b.get('point1'), a.get('point1'), left.get('_bounds'), radius=100))
        if action == 'DRAG':
            return (point_matches(b.get('point1'), a.get('point1'), left.get('_bounds'))
                    and point_matches(b.get('point2'), a.get('point2')))
        if action == 'WAIT':
            return True  # Duration is motor timing; native success still gates reward.
        return a == b

    matched = len(reference) == len(candidate) and all(equivalent(a, b) for a, b in zip(reference, candidate))
    return {'status': 'match' if matched else 'different', 'match': matched,
            'fallback_steps': sum(bool(s.get('fallback')) for s in reference)}
