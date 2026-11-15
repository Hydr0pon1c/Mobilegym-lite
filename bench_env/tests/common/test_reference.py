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

from copy import deepcopy
import pytest

from bench_env.reference import binary_rewards, compare_paths, created_entities, operation_token
from bench_env.reference import practical_tokens, compare_practical_tokens
from bench_env.reference import created_provider_entities


def click(step, entity, point=(100, 200)):
    return dict(source_step=step, action_type='CLICK', action_data={'point': list(point)},
        route={'app': 'demo', 'path': '/people'},
        before={'target': {'unique': True, 'disabled': False,
                          'identity': {'kind': 'action', 'id': 'follow', 'gesture': 'tap',
                                       'params': {'userId': entity}}}})


def test_same_target_accepts_coordinate_variation_but_not_another_entity():
    reference = [click(1, 'alice')]
    assert compare_paths(reference, [click(1, 'alice', (800, 200))])['match']
    assert not compare_paths(reference, [click(1, 'bob')])['match']


def test_productive_order_and_extra_branches_are_preserved():
    reference = [click(1, 'alice'), click(2, 'bob')]
    assert not compare_paths(reference, reference[::-1])['match']
    assert not compare_paths(reference, [*reference, click(3, 'charlie')])['match']


def test_unresolved_is_excluded_instead_of_becoming_a_training_negative():
    ambiguous = click(1, 'alice')
    ambiguous['before']['target']['unique'] = False
    comparison = compare_paths([ambiguous], [ambiguous])
    assert comparison['status'] == 'unresolved'
    assert binary_rewards(True, comparison) == {'native': 1, 'restricted': None, 'eligible': False}


def test_native_success_is_required_even_when_actions_match():
    comparison = compare_paths([click(1, 'alice')], [click(1, 'alice')])
    assert binary_rewards(False, comparison)['restricted'] == 0
    assert binary_rewards(True, comparison)['restricted'] == 1
    assert binary_rewards(True, {'status': 'different', 'match': False})['restricted'] == 0


def test_no_implicit_wait_removal_and_only_explicit_cleaning_is_applied():
    reference = [click(1, 'alice')]
    candidate = [dict(source_step=2, action_type='WAIT', action_data={'value': 2}, route={}), *reference]
    assert compare_paths(reference, candidate)['status'] == 'unresolved'
    assert compare_paths(reference, candidate, ignored_candidate={2})['match']


def test_click_does_not_match_a_long_press_binding():
    wrong_gesture = click(1, 'alice')
    wrong_gesture['before']['target']['identity']['gesture'] = 'longPress'
    assert operation_token(wrong_gesture) is None


def test_terminal_prose_does_not_replace_task_judging():
    first = dict(source_step=1, action_type='ANSWER', action_data={'value': 'first'}, route={})
    second = deepcopy(first)
    second['action_data']['value'] = 'second'
    comparison = compare_paths([first], [second])
    assert comparison['match']
    assert binary_rewards(False, comparison)['restricted'] == 0


def test_type_keeps_payload_and_clear_semantics():
    first = click(1, 'field')
    first.update(action_type='TYPE', action_data={'value': 'text', 'clear': False})
    second = deepcopy(first)
    second['action_data']['clear'] = True
    assert not compare_paths([first], [second])['match']


def test_new_ids_are_bound_but_existing_entities_are_not_aliased():
    before = {'notes': {'notes': [{'id': 'existing'}]}}
    after = {'notes': {'notes': [{'id': 'new_a'}, {'id': 'existing'}]}}
    entities = created_entities(before, after)
    assert entities == [{'collection': ['notes', 'notes'], 'id': 'new_a'}]
    left, right = click(1, 'alice'), click(1, 'alice')
    left['created_entities'] = entities
    right['created_entities'] = [{'collection': ['notes', 'notes'], 'id': 'new_b'}]
    left['route'] = {'app': 'notes', 'path': '/note/new_a'}
    right['route'] = {'app': 'notes', 'path': '/note/new_b'}
    assert compare_paths([left], [right])['match']
    left['created_entities'] = right['created_entities'] = []
    assert not compare_paths([left], [right])['match']


def test_lazily_opened_app_data_is_not_assumed_to_be_newly_created():
    assert created_entities({}, {'other': {'items': [{'id': 'existing'}]}}) == []


def test_duplicate_tags_need_a_unique_row_witness():
    first = click(1, 'alice')
    target = first['before']['target']
    target.update(unique=False, label='Alice')
    first['before']['controls'] = [deepcopy(target), {**deepcopy(target), 'label': 'Bob'}]
    assert operation_token(first)['target']['row_label'] == 'Alice'
    first['before']['controls'].append(deepcopy(target))
    assert operation_token(first) is None


def test_binary_reward_rejects_fractional_progress():
    with pytest.raises(TypeError, match='strict boolean'):
        binary_rewards(0.5, {'status': 'match', 'match': True})


def test_scroll_chunks_merge_but_opposite_direction_is_a_different_operation():
    def scroll(number, delta):
        row = click(number, 'unused')
        row.update(action_type='SWIPE', next_route=row['route'], app_changed_paths=[])
        row['before']['ui'] = {'scrolls': [{'id': 'list', 'top': 100, 'left': 0}]}
        row['after'] = {'ui': {'scrolls': [{'id': 'list', 'top': 100 + delta, 'left': 0}]}}
        return row
    reference = [scroll(1, 50), click(2, 'alice')]
    assert compare_paths(reference, [scroll(1, 20), scroll(3, 30), click(2, 'alice')])['match']
    assert not compare_paths(reference, [scroll(1, 50), scroll(3, -30), click(2, 'alice')])['match']


def test_drag_requires_the_same_resulting_value():
    first = click(1, 'slider')
    first.update(action_type='DRAG', app_effects=[{'path': ['settings', 'volume'], 'after': 2}])
    second = deepcopy(first)
    second['app_effects'][0]['after'] = 8
    assert not compare_paths([first], [second])['match']


def test_desktop_route_can_be_absent():
    step = dict(source_step=1, action_type='AWAKE', action_data={'value': 'demo'}, route=None)
    assert operation_token(step)['route'] == {'app': None, 'path': '/'}


def test_practical_fallback_has_bounds_and_is_not_a_candidate_wildcard():
    gold = click(1, 'alice')
    gold['before']['target'] = None
    moved = click(1, 'alice', (115, 200))
    assert compare_practical_tokens(practical_tokens([gold]), practical_tokens([moved]))['match']
    assert not compare_practical_tokens(practical_tokens([gold]), practical_tokens([click(1, 'bob', (300, 200))]))['match']
    assert not compare_practical_tokens(practical_tokens([moved]), practical_tokens([gold]))['match']


def test_practical_cleaning_does_not_drop_a_visible_change_or_mutation():
    gold = click(1, 'alice')
    noop = click(2, 'bob')
    noop['noop_candidate'] = True
    assert compare_practical_tokens(practical_tokens([gold]), practical_tokens([noop, gold]))['match']
    noop['noop_candidate'] = False
    assert not compare_practical_tokens(practical_tokens([gold]), practical_tokens([noop, gold]))['match']


def test_fallback_type_preserves_payload_and_clear():
    first = click(1, 'field')
    first['before']['target'] = None
    first.update(action_type='TYPE', action_data={'point': [100, 200], 'value': 'hello', 'clear': True})
    second = deepcopy(first)
    second['action_data']['value'] = 'bye'
    assert not compare_practical_tokens(practical_tokens([first]), practical_tokens([second]))['match']


def test_created_sms_ids_bind_from_os_provider_state():
    before = {'os': {'providers': {'sms': {'conversations': []}}}}
    after = {'os': {'providers': {'sms': {'conversations': [{'id': 'conv_new'}]}}}}
    entities = created_provider_entities(before, after)
    assert entities == [{'collection': ['@providers', 'sms', 'conversations'], 'id': 'conv_new'}]
    assert created_provider_entities({}, after) == []  # Loading old provider data is not creation.


def test_recorded_motor_mode_keeps_scroll_count_without_reference_ui_evidence():
    raw, observed = [], []
    for i in (1, 2):
        row = click(i, 'unused')
        row.update(action_type='SWIPE', action_data={'point1': [500, 750], 'point2': [500, 250]},
                   next_route=row['route'], app_changed_paths=['unobserved'])
        row['before']['target'] = None
        raw.append(row)
        actual = deepcopy(row)
        actual['app_changed_paths'] = []
        actual['before']['ui'] = {'scrolls': [{'id': 'list', 'top': i * 50, 'left': 0}]}
        actual['after'] = {'ui': {'scrolls': [{'id': 'list', 'top': (i + 1) * 50, 'left': 0}]}}
        observed.append(actual)
    assert compare_practical_tokens(practical_tokens(raw, clean=False), practical_tokens(observed, clean=False))['match']
    assert not compare_practical_tokens(practical_tokens(raw), practical_tokens(observed))['match']
