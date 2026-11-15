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

"""Coordinator handshake and version scoping, without accelerator imports."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


PATH = Path(__file__).resolve().parents[1] / "cookbooks/mobilegym/serve_vllm.py"
spec = spec_from_file_location("mobilegym_serve_vllm_test", PATH)
serve = module_from_spec(spec)
spec.loader.exec_module(serve)


@pytest.mark.parametrize("outcome", ["ready", "timeout", "exit", "eof"])
def test_coordinator_handshake(monkeypatch, outcome):
    pipe = Mock()
    process = SimpleNamespace(sentinel=object(), pid=123)
    addresses = ("ipc://front", "ipc://back", "ipc://publish")
    pipe.recv.return_value = addresses
    if outcome == "eof":
        pipe.recv.side_effect = EOFError
    ready = [] if outcome == "timeout" else [process.sentinel] if outcome == "exit" else [pipe]
    wait = Mock(return_value=ready)
    monkeypatch.setattr(serve.multiprocessing.connection, "wait", wait)

    if outcome == "ready":
        assert serve._wait_for_zmq_addrs(SimpleNamespace(proc=process), pipe) == addresses
    else:
        message = {"timeout": "120 seconds", "exit": "exited", "eof": "failed"}[outcome]
        with pytest.raises(RuntimeError, match=message):
            serve._wait_for_zmq_addrs(SimpleNamespace(proc=process), pipe)
    wait.assert_called_once_with([pipe, process.sentinel], timeout=120)
    pipe.close.assert_called_once()
    if outcome in {"timeout", "exit"}:
        pipe.recv.assert_not_called()


@pytest.mark.parametrize("version", ["0.19.0", "0.19.0+ppu", "0.20.1"])
def test_cli_patch_is_scoped_to_pinned_version(monkeypatch, version):
    original = object()
    coordinator = SimpleNamespace(_wait_for_zmq_addrs=original)
    cli = Mock()
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.main", SimpleNamespace(main=cli))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.coordinator", SimpleNamespace(DPCoordinator=coordinator))
    monkeypatch.setattr(serve, "version", lambda name: version)

    serve.main()

    cli.assert_called_once_with()
    expected = serve._wait_for_zmq_addrs if version.startswith("0.19.0") else original
    assert coordinator._wait_for_zmq_addrs is expected
