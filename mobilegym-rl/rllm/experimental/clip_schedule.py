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

"""Optional outer-step clipping schedule, independent of optimizer update counts."""
from collections.abc import Mapping
from dataclasses import replace
import math


def validate_clip_schedule(schedule, total_steps=None):
    if schedule is None:
        return None
    required = {"switch_step", "clip_ratio_low", "clip_ratio_high"}
    if not isinstance(schedule, Mapping) or set(schedule) != required:
        raise ValueError(f"actor_clip_schedule requires exactly {sorted(required)}")
    step = schedule["switch_step"]
    if type(step) is not int or step < 2:
        raise ValueError("actor_clip_schedule.switch_step must be an integer >= 2")
    if total_steps is not None and step > total_steps:
        raise ValueError("actor_clip_schedule switches after the final training step")
    bounds = {}
    for key in ("clip_ratio_low", "clip_ratio_high"):
        value = schedule[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"actor_clip_schedule.{key} must be positive and finite")
        bounds[key] = float(value)
    if bounds["clip_ratio_low"] >= 1:
        raise ValueError("actor_clip_schedule.clip_ratio_low must be < 1")
    return {"switch_step": step, **bounds}


def scheduled_actor_config(config, schedule, global_step):
    """Copy only clip bounds; preserve the original config and all loss normalization."""
    if schedule is None:
        return config
    if type(global_step) is not int or global_step < 1:
        raise ValueError("Scheduled actor loss requires the positive outer actor_global_step")
    if global_step < schedule["switch_step"]:
        return config
    return replace(config, clip_ratio_low=schedule["clip_ratio_low"],
                   clip_ratio_high=schedule["clip_ratio_high"])
