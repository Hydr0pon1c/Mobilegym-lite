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

import { describe, expect, it } from 'vitest';

import { defaultWmrVarExpression } from '../os/wmr/engine/parser';
import { VarContext } from '../os/wmr/engine/variables';
import type { WmrVar } from '../os/wmr/engine/types';

describe('WMR variable defaults', () => {
  it('initializes expressionless string vars as empty strings', () => {
    const nodes: WmrVar[] = [
      {
        tag: 'Var',
        name: 'alert_type',
        type: 'string',
        expression: defaultWmrVarExpression('string'),
        const: true,
      },
      {
        tag: 'Var',
        name: 'frame',
        type: 'number',
        expression: defaultWmrVarExpression('number'),
        const: true,
      },
    ];
    const vars = new VarContext('test-string-var-default');

    vars.initFromNodes(nodes);

    expect(vars.get('alert_type')).toBe('');
    expect(vars.get('frame')).toBe(0);
  });

  it('keeps an explicit string expression', () => {
    const vars = new VarContext('test-string-var-expression');

    vars.initFromNodes([{
      tag: 'Var',
      name: 'alert_type',
      type: 'string',
      expression: "'rain_alert'",
    }]);

    expect(vars.get('alert_type')).toBe('rain_alert');
  });
});
