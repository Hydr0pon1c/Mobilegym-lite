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

import ctypes
import gc
from types import SimpleNamespace

from rllm.experimental import memory_diagnostics


def test_trim_host_memory_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(memory_diagnostics, "_TRIM_ENABLED", False)
    monkeypatch.setattr(gc, "collect", lambda: (_ for _ in ()).throw(AssertionError("unexpected gc")))

    assert memory_diagnostics.trim_host_memory() is False


def test_trim_host_memory_collects_and_calls_glibc(monkeypatch):
    calls = []

    def fake_collect():
        calls.append("gc")

    def fake_malloc_trim(pad):
        calls.append(("malloc_trim", pad))
        return 1

    monkeypatch.setattr(memory_diagnostics, "_TRIM_ENABLED", True)
    monkeypatch.setattr(gc, "collect", fake_collect)
    monkeypatch.setattr(ctypes, "CDLL", lambda _name: SimpleNamespace(malloc_trim=fake_malloc_trim))

    assert memory_diagnostics.trim_host_memory() is True
    assert calls == ["gc", ("malloc_trim", 0)]
