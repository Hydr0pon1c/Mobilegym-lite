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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbooks.mobilegym import check_training_inputs as gate


def identity():
    root = Path(gate.__file__).resolve().parents[2]
    return {"sources": {"rllm": str(root / "rllm/__init__.py"),
                        "verl": str(root / "verl/verl/__init__.py")}}


def test_gate_records_success_and_failure_without_stale_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_VERL_ROOT", raising=False)
    monkeypatch.setattr(gate, "runtime_identity", identity)
    monkeypatch.setattr(gate, "check_inputs", lambda _: {"checks": ["test"]})
    output = tmp_path / "report.json"
    assert gate.run_check(tmp_path, output)["passed"]

    def fail(_):
        raise RuntimeError("variable prompt regression")

    monkeypatch.setattr(gate, "check_inputs", fail)
    with pytest.raises(RuntimeError, match="variable prompt"):
        gate.run_check(tmp_path, output)
    saved = json.loads(output.read_text())
    assert not saved["passed"]
    assert saved["error"] == "RuntimeError: variable prompt regression"


def test_gate_rejects_shadowed_verl_before_processor(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_VERL_ROOT", raising=False)
    runtime = identity()
    runtime["sources"]["verl"] = "/unrelated/verl/__init__.py"
    monkeypatch.setattr(gate, "runtime_identity", lambda: runtime)
    monkeypatch.setattr(gate, "check_inputs", lambda _: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="Unexpected verl"):
        gate.run_check(tmp_path, tmp_path / "report.json")


@pytest.mark.parametrize("visual", [False, True])
def test_native_and_visual_train_use_the_same_gate(tmp_path, monkeypatch, visual):
    from omegaconf import OmegaConf
    from cookbooks.mobilegym.train import _check_training_inputs

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    config = OmegaConf.create({"actor_rollout_ref": {"model": {"path": str(model)}},
                              "train_visual_config": "visual" if visual else None})
    runtime = SimpleNamespace(runs_dir=str(tmp_path / "logs"), run_name="test")
    calls = []
    monkeypatch.setattr(gate, "run_check", lambda *args: calls.append(args))
    _check_training_inputs(config, runtime)
    assert calls == [(model, tmp_path / "logs/test/training-input-check.json")]

    def fail(*args):
        raise RuntimeError("bad inputs")

    monkeypatch.setattr(gate, "run_check", fail)
    with pytest.raises(RuntimeError, match="bad inputs"):
        _check_training_inputs(config, runtime)
