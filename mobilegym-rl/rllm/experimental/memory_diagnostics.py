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

"""Low-overhead, opt-in host-memory diagnostics for RL training.

Enable with ``RLLM_MEMORY_DIAGNOSTICS=1``. The helpers intentionally avoid
``gc.get_objects()`` and do not move or copy tensors, so enabling diagnostics
does not change the training graph or allocator lifecycle.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any


_ENABLED = os.environ.get("RLLM_MEMORY_DIAGNOSTICS", "").lower() in {"1", "true", "yes", "on"}
_TRIM_ENABLED = os.environ.get("RLLM_TRIM_HOST_MEMORY", "").lower() in {"1", "true", "yes", "on"}
_CGROUP_MEMORY_DIR = "/sys/fs/cgroup/memory"


def enabled() -> bool:
    return _ENABLED


def trim_host_memory() -> bool:
    """Release unused Python/glibc arenas at a training-batch boundary.

    MobileGym decodes screenshots concurrently.  On large-CPU machines glibc
    may retain many per-thread heaps after a batch is discarded, and Ray counts
    those resident pages toward its node OOM threshold.  This opt-in helper is
    called only after ``TrainerState.reset_batch()`` has dropped the previous
    batch.  It never touches live tensors.

    Returns whether glibc reported that it released any pages.
    """
    if not _TRIM_ENABLED:
        return False

    # Import lazily so the disabled path remains effectively free and works on
    # non-glibc platforms where malloc_trim is unavailable.
    import ctypes
    import gc

    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except (AttributeError, OSError):
        return False


def _read_int(path: str) -> int | None:
    try:
        with open(path) as stream:
            return int(stream.read().strip())
    except (OSError, ValueError):
        return None


def _read_key_values(path: str) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(path) as stream:
            for line in stream:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        values[fields[0].removesuffix(":")] = int(fields[1])
                    except ValueError:
                        continue
    except OSError:
        pass
    return values


def _process_memory() -> dict[str, int]:
    status = _read_key_values("/proc/self/status")
    smaps = _read_key_values("/proc/self/smaps_rollup")
    result: dict[str, int] = {}
    # /proc status and smaps values are expressed in KiB.
    for key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSize"):
        if key in status:
            result[f"process_{key}_bytes"] = status[key] * 1024
    for key in ("Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Private_Dirty"):
        if key in smaps:
            result[f"process_{key}_bytes"] = smaps[key] * 1024
    return result


def _cgroup_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    for filename in ("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt"):
        value = _read_int(os.path.join(_CGROUP_MEMORY_DIR, filename))
        if value is not None:
            key = filename.removeprefix("memory.").replace(".", "_")
            result[f"cgroup_{key}"] = value
    stats = _read_key_values(os.path.join(_CGROUP_MEMORY_DIR, "memory.stat"))
    for key in ("total_rss", "total_cache", "total_shmem", "total_mapped_file"):
        if key in stats:
            result[f"cgroup_{key}_bytes"] = stats[key]
    oom = _read_key_values(os.path.join(_CGROUP_MEMORY_DIR, "memory.oom_control"))
    if "oom_kill" in oom:
        result["cgroup_oom_kill"] = oom["oom_kill"]
    return result


def _tensor_nbytes(value: Any) -> int | None:
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if not callable(numel) or not callable(element_size):
        return None
    try:
        return int(numel()) * int(element_size())
    except Exception:  # noqa: BLE001 - diagnostics must never break training
        return None


def _mapping_tensor_bytes(value: Any, seen: set[int]) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        return 0, 0
    count = 0
    total = 0
    for item in value.values():
        item_id = id(item)
        if item_id in seen:
            continue
        size = _tensor_nbytes(item)
        if size is not None:
            seen.add(item_id)
            count += 1
            total += size
    return count, total


def _string_bytes(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_string_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_string_bytes(item) for item in value)
    return 0


def _episode_stats(episodes: Any) -> dict[str, int]:
    if episodes is None:
        return {}
    if not isinstance(episodes, (list, tuple)):
        episodes = [episodes]

    stats = {
        "episodes": len(episodes),
        "trajectories": 0,
        "steps": 0,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "chat_string_bytes": 0,
        "multimodal_tensors": 0,
        "multimodal_tensor_bytes": 0,
    }
    seen_tensors: set[int] = set()
    for episode in episodes:
        trajectories = getattr(episode, "trajectories", None) or []
        stats["trajectories"] += len(trajectories)
        for trajectory in trajectories:
            steps = getattr(trajectory, "steps", None) or []
            stats["steps"] += len(steps)
            for step in steps:
                output = getattr(step, "model_output", None)
                if output is not None:
                    stats["prompt_tokens"] += len(getattr(output, "prompt_ids", None) or [])
                    stats["response_tokens"] += len(getattr(output, "completion_ids", None) or [])
                    count, size = _mapping_tensor_bytes(getattr(output, "multi_modal_inputs", None), seen_tensors)
                    stats["multimodal_tensors"] += count
                    stats["multimodal_tensor_bytes"] += size
                stats["chat_string_bytes"] += _string_bytes(getattr(step, "chat_completions", None))
    return stats


def _backend_batch_stats(backend_batch: Any) -> dict[str, int]:
    if backend_batch is None:
        return {}
    result = {
        "batch_tensors": 0,
        "batch_tensor_bytes": 0,
        "batch_multimodal_tensors": 0,
        "batch_multimodal_tensor_bytes": 0,
    }
    seen_tensors: set[int] = set()

    batch = getattr(backend_batch, "batch", None)
    if batch is not None:
        try:
            items = batch.items()
        except Exception:  # noqa: BLE001
            items = []
        for _, value in items:
            value_id = id(value)
            if value_id in seen_tensors:
                continue
            size = _tensor_nbytes(value)
            if size is not None:
                seen_tensors.add(value_id)
                result["batch_tensors"] += 1
                result["batch_tensor_bytes"] += size

    non_tensor_batch = getattr(backend_batch, "non_tensor_batch", None)
    if isinstance(non_tensor_batch, Mapping):
        mm_collection = non_tensor_batch.get("multi_modal_inputs")
        if mm_collection is None:
            mm_collection = []
        for mm_inputs in mm_collection:
            count, size = _mapping_tensor_bytes(mm_inputs, seen_tensors)
            result["batch_multimodal_tensors"] += count
            result["batch_multimodal_tensor_bytes"] += size
    return result


def mark(
    stage: str,
    *,
    trainer_state: Any = None,
    episode: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Emit one JSON diagnostic record when diagnostics are enabled."""
    if not _ENABLED:
        return
    try:
        record: dict[str, Any] = {
            "time": time.time(),
            "stage": stage,
            "pid": os.getpid(),
            **_process_memory(),
            **_cgroup_memory(),
        }
        if trainer_state is not None:
            record["global_step"] = getattr(trainer_state, "global_step", None)
            record["epoch"] = getattr(trainer_state, "epoch", None)
            record.update(_episode_stats(getattr(trainer_state, "episodes", None)))
            record.update(_backend_batch_stats(getattr(trainer_state, "backend_batch", None)))
        elif episode is not None:
            record.update(_episode_stats(episode))
        if extra:
            record.update(extra)
        print(f"[RLLM_MEM] {json.dumps(record, sort_keys=True, default=str)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break training
        print(f"[RLLM_MEM] diagnostic_error stage={stage!r} error={exc!r}", flush=True)
