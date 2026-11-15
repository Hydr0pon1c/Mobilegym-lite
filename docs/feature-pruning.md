# Feature0 and MobileGym-Lite

Feature pruning is an episode-local intervention on available app functions.
Both released configurations keep the original task data and native judge.
They do not delete source files, assets, task templates, or resource records.

## Configurations and scope

`feature0` disables every reviewed dispensable module for the current template.
The reviewed set covers 78 training templates and 127 template/module assignments
across Clock, Notes, Calendar, and WeChat. Membership is explicit in
[`feature_completeness.py`](../bench_env/feature_completeness.py); it does not
consult sampled task parameters, answers, observed reference paths, or RNG.

| Profile | Modules eligible for disabling |
| --- | --- |
| `clock_alarm` | World clock, stopwatch, timer |
| `clock_world` | Stopwatch, timer |
| `notes_note` | To-do list |
| `notes_todo` | Notes settings |
| `calendar_fixed` | Date calculator |
| `calendar_dates` | Calendar settings |
| `wechat_communication` | Moments, profile editing |
| `wechat_personal` | Moments |
| `wechat_social` | Profile editing |

Cross-app tasks combine their applicable profiles. Conditional tasks retain
required functions for both branches. Alternate entrances and shared helpers,
including WeChat chat media and QR functions, remain available. Unknown templates
have no feature mask and run in Full.

`feature0_selective_v1` starts from the same candidate masks and restores:

| Profile | Restored module |
| --- | --- |
| `clock_alarm` | `clock.world` |
| `calendar_dates` | `calendar.settings` |
| `wechat_communication` | `wechat.profile_edit` |

This leaves 72 effective templates and 87 disabled template/module assignments.
This configuration is **MobileGym-Lite**; `lite_v1` is a separate internal
resource-plus-feature compatibility mode.

## Inspect masks without a browser

From the repository root:

```python
from types import SimpleNamespace
from bench_env.feature_completeness import feature_plan

example = SimpleNamespace(id="clock.ToggleAlarm")
print(feature_plan(example, "feature0"))
print(feature_plan(example, "feature0_selective_v1"))
```

Inspect the complete task membership in `PROFILE_TASKS`, and the restorations in
`SELECTIVE_RESTORE`. `feature_plan()` normalizes sampled instance suffixes to
base template IDs.

## Training and evaluation

Use `configs/training/feature0.json` or `feature0-selective.json` through
`TRAIN_CONFIG`. Both set `val_completeness=null`. Full uses `full.json`. Compare
conditions with the same hardware, runtime, task sampling, reward, clip settings,
and frontend build.

For a diagnostic evaluation in a pruned environment, pass
`--completeness feature0` or `--completeness feature0_selective_v1` to
`python -m bench_env.run`. Omit `--completeness` for standard Full evaluation.
Availability plans are environment metadata, not policy-prompt content.

## Runtime contract

[`BaseRunner`](../bench_env/runner/base.py) resets and prepares the task before
[`apply_completeness()`](../bench_env/completeness.py) applies the plan. The
frontend API must report `features_v1`; mismatched builds fail instead of silently
using Full. App UI gates and [`FeatureRoutes`](../os/featureAvailability.tsx)
remove entrances and reject direct/deep-link routes before mounting unavailable
pages. Apps consume module IDs, not task IDs. Masks reset between episodes and
remain isolated between browser pages.

## Checks

```bash
python -m pytest bench_env/tests/common/test_feature_completeness.py -q
npm test -- tests/feature_availability.test.ts
python -m pytest bench_env/tests/common/test_feature_completeness_live.py \
  -m live --sim-url http://127.0.0.1:3000 -q
```

The tests cover mask scope, required dependencies, reset/isolation, route denial,
and restored functional controls.
