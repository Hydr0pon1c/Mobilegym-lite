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

"""Private, in-memory reference collection for terminal binary RL rewards."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .reference import (DOM_PROBE, changed_paths, compare_practical_tokens,
                        created_entities, created_provider_entities, practical_tokens)


@lru_cache(maxsize=4)
def load_reference_bundle(path: str) -> dict:
    bundle = json.loads(Path(path).read_text())
    if bundle.get('schema_version') != 2 or bundle.get('profile') != 'practical_v1':
        raise ValueError('Expected a schema 2 / practical_v1 reference bundle')
    bundle['by_task'] = {row['task_id']: row for row in bundle['tasks']}
    if len(bundle['by_task']) != len(bundle['tasks']):
        raise ValueError('Duplicate reference task IDs')
    return bundle


def reference_entry(bundle: dict, row: dict) -> dict:
    entry = bundle['by_task'].get(row['task_id'])
    if entry is None or row.get('task_seed') != entry['task_seed']:
        raise ValueError(f'Reference instance mismatch: {row["task_id"]}; use the exported train.samples.jsonl')
    return entry


def reference_url(url: str, bundle: dict) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    expected = str(bundle['time_ms'])
    if query.get('benchmark_time_ms', expected) != expected:
        raise ValueError('Environment clock conflicts with the reference bundle')
    query['benchmark_time_ms'] = expected
    return urlunsplit(parts._replace(query=urlencode(query)))


class ReferenceTracker:
    """Attach for one acquired environment; never mask actions or stop divergence.

    Full states and screenshots are not persisted. Only operation evidence is kept
    until terminal comparison. The caller must detach before releasing the page.
    """

    def __init__(self, env, entry: dict, time_ms: int):
        self.env, self.entry, self.time_ms = env, entry, time_ms
        self.steps = []
        self.overhead_s = 0.0
        self.original_step = env.step

    def attach(self):
        self.env.step = self.step

    def detach(self):
        self.env.step = self.original_step

    async def setup_observer(self, env, task, observation):
        clock = await env.page.evaluate('() => ({now: __SIM_TIME__.now(), config: __SIM_TIME__.getConfig()})')
        if clock['now'] != self.time_ms or clock['config'].get('flowing') is not False:
            raise ValueError('Reference reward requires a frontend with benchmark_time_ms at boot')
        if task.description != self.entry['task_name'] or task._seed != self.entry['task_seed']:
            raise ValueError('Sampled task differs from the reference instance')

    async def snapshot(self, action):
        point = action.data.get('point') or action.data.get('point1')
        css = list(self.env._p2c(*self.env._parse_point(point))) if point else None
        return await self.env.page.evaluate(DOM_PROBE, dict(point=css, gesture=action.action_type.value, probes=False))

    async def step(self, action):
        start = perf_counter()
        before = await self.snapshot(action)
        full_before = await self.env.get_state()
        state_before = full_before.get('apps', {})
        route_before = await self.env._get_route()
        self.overhead_s += perf_counter() - start
        result = await self.original_step(action)
        start = perf_counter()
        after = await self.snapshot(action)
        full_after = await self.env.get_state()
        state_after = full_after.get('apps', {})
        route_after = await self.env._get_route()
        differences = changed_paths(state_before, state_after)
        def effects(left, right, path=()):
            if isinstance(left, dict) and isinstance(right, dict):
                return [item for k in sorted(left.keys() | right.keys())
                        for item in effects(left.get(k), right.get(k), (*path, k))]
            return [] if left == right else [{'path': list(path), 'after': right}]
        # Retain only fields consumed by canonicalization/cleaning. No policy input.
        target = before['target']
        controls = [c for c in before['controls'] if target and not target['unique']
                    and c['identity'] == target['identity'] and c.get('label') == target.get('label')]
        compact_before = {k: before[k] for k in ('target', 'viewport', 'ui')}
        compact_before['controls'] = controls
        self.steps.append(dict(source_step=len(self.steps) + 1, action_type=action.action_type.value,
            action_data=deepcopy(action.data), route=route_before, next_route=route_after,
            before=compact_before,
            after={'ui': after['ui']}, created_entities=created_entities(state_before, state_after)
                + created_provider_entities(full_before, full_after),
            app_changed_paths=differences,
            app_effects=effects(state_before, state_after) if action.action_type.value == 'DRAG' else [],
            noop_candidate=not differences and before['ui'] == after['ui'] and route_before == route_after))
        self.overhead_s += perf_counter() - start
        return result

    def finish(self):
        operations = practical_tokens(self.steps, clean=self.entry.get('clean', True))
        return {**compare_practical_tokens(self.entry['operations'], operations),
                'tier': self.entry['tier'], 'operations': operations,
                'reference_task_id': self.entry['task_id'], 'overhead_s': self.overhead_s,
                'steps': len(self.steps)}
