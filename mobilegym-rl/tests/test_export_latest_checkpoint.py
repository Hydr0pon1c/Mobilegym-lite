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
from argparse import Namespace
import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cookbooks.mobilegym.export_latest_checkpoint import (
    ensure_lora_metadata,
    export,
    parse_args,
    patch_adapter_config,
    resolve_latest_actor,
    update_latest_link,
    validate_export,
)


def _tiny_export(root: Path, dtype_key="dtype", tensor_dtype="BF16", data=b"\x00\x00"):
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3VLForConditionalGeneration"], dtype_key: "bfloat16",
    }))
    header = json.dumps({"layer.weight": {
        "dtype": tensor_dtype, "shape": [1], "data_offsets": [0, len(data)],
    }}).encode()
    header += b" " * (-len(header) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)


@pytest.mark.parametrize("dtype_key", ["dtype", "torch_dtype"])
def test_validate_export_needs_no_transformers_import(tmp_path, monkeypatch, dtype_key):
    _tiny_export(tmp_path, dtype_key=dtype_key)
    # Simulate an unavailable dependency scan, independently of import order.
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = validate_export(tmp_path, "bfloat16")
    assert result == {
        "architecture": "Qwen3VLForConditionalGeneration", "dtype": "torch.bfloat16",
        "weight_files": 1, "tensor_keys": 1, "weight_dtype_counts": {"BF16": 1},
    }


def test_validate_export_still_rejects_wrong_weight_dtype(tmp_path):
    _tiny_export(tmp_path, tensor_dtype="F32", data=b"\x00" * 4)
    with pytest.raises(ValueError, match="outside BF16"):
        validate_export(tmp_path, "bfloat16")


def test_validate_export_still_checks_tensor_payload_size(tmp_path):
    from safetensors import SafetensorError

    _tiny_export(tmp_path)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1])
    with pytest.raises(SafetensorError):
        validate_export(tmp_path, "bfloat16")


def _actor_checkpoint(root: Path, step: int) -> Path:
    actor = root / f"global_step_{step}" / "actor"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "fsdp_config.json").write_text('{"world_size": 2}\n', encoding="utf-8")
    (actor / "huggingface" / "config.json").write_text("{}\n", encoding="utf-8")
    (actor / "model_world_size_2_rank_0.pt").touch()
    return actor


def test_cli_defaults_to_full_parameter_export(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_latest_checkpoint.py",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-dir",
            str(tmp_path / "models"),
        ],
    )

    args = parse_args()

    assert args.lora_rank == 0
    assert args.lora_alpha == 1


def test_resolve_latest_actor_uses_checkpoint_marker(tmp_path: Path):
    _actor_checkpoint(tmp_path, 5)
    expected = _actor_checkpoint(tmp_path, 10)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("10\n", encoding="utf-8")

    step, actor = resolve_latest_actor(tmp_path)

    assert step == 10
    assert actor == expected.resolve()


def test_resolve_latest_actor_falls_back_to_highest_step(tmp_path: Path):
    _actor_checkpoint(tmp_path, 5)
    expected = _actor_checkpoint(tmp_path, 15)

    step, actor = resolve_latest_actor(tmp_path)

    assert step == 15
    assert actor == expected.resolve()


def test_lora_metadata_is_written_and_mismatch_is_rejected(tmp_path: Path):
    actor = _actor_checkpoint(tmp_path, 5)
    ensure_lora_metadata(actor, rank=16, alpha=16, exclude_modules=r".*visual.*")

    metadata = json.loads((actor / "lora_train_meta.json").read_text(encoding="utf-8"))
    assert metadata == {
        "r": 16,
        "lora_alpha": 16,
        "task_type": "CAUSAL_LM",
        "exclude_modules": r".*visual.*",
    }

    with pytest.raises(ValueError, match="LoRA metadata mismatch"):
        ensure_lora_metadata(actor, rank=32, alpha=16, exclude_modules=r".*visual.*")


def test_patch_adapter_config_preserves_targets_and_sets_training_metadata(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config_path = adapter / "adapter_config.json"
    config_path.write_text(
        json.dumps({"r": 16, "lora_alpha": 0, "target_modules": ["q_proj", "v_proj"]}),
        encoding="utf-8",
    )

    patch_adapter_config(adapter, rank=16, alpha=32, exclude_modules=r".*visual.*")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["r"] == 16
    assert config["lora_alpha"] == 32
    assert config["task_type"] == "CAUSAL_LM"
    assert config["exclude_modules"] == r".*visual.*"
    assert config["target_modules"] == ["q_proj", "v_proj"]


def test_update_latest_link_is_relative_and_atomic(tmp_path: Path):
    final = tmp_path / "global_step_20"
    final.mkdir()

    update_latest_link(tmp_path, final)

    latest = tmp_path / "latest"
    assert latest.is_symlink()
    assert latest.readlink() == Path("global_step_20")
    assert latest.resolve() == final.resolve()


@pytest.fixture
def full_export(tmp_path: Path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    actor = _actor_checkpoint(checkpoints, 8)
    (checkpoints / "latest_checkpointed_iteration.txt").write_text("8\n", encoding="utf-8")
    output = tmp_path / "models"
    calls: list[tuple[Path, Path]] = []

    def fake_fsdp_merge(actor_dir: Path, merged_dir: Path) -> None:
        calls.append((actor_dir, merged_dir))
        merged_dir.mkdir()

    monkeypatch.setattr(
        "cookbooks.mobilegym.export_latest_checkpoint.run_verl_fsdp_merge",
        fake_fsdp_merge,
    )
    monkeypatch.setattr(
        "cookbooks.mobilegym.export_latest_checkpoint.validate_export",
        lambda model_dir, expected_dtype: {
            "architecture": "Qwen3VLForConditionalGeneration",
            "dtype": expected_dtype,
            "weight_files": 1,
            "tensor_keys": 1,
            "weight_dtype_counts": {"BF16": 1},
        },
    )

    args = Namespace(
        checkpoint_dir=checkpoints,
        adapter_dir=None,
        base_model=None,
        global_step=None,
        output_dir=output,
        lora_rank=0,
        lora_alpha=16,
        lora_exclude_modules="null",
        device="cpu",
        max_shard_size="5GB",
        merge_dtype="float32",
        output_dtype="bfloat16",
    )
    return args, actor, calls


def test_full_parameter_export_skips_lora_merge(full_export):
    args, actor, calls = full_export
    final = export(args)

    assert len(calls) == 1
    assert calls[0][0] == actor.resolve()
    assert calls[0][1].name == "merged"
    assert calls[0][1].parent.parent == args.output_dir.resolve()
    metadata = json.loads((final / "mobilegym_export.json").read_text(encoding="utf-8"))
    assert metadata["training_mode"] == "full"
    assert metadata["lora_rank"] == 0
    assert str(actor / "model_world_size_2_rank_0.pt") in metadata["source_files"]
    assert (args.output_dir / "latest").resolve() == final.resolve()


def test_unchanged_export_is_reused_without_merging(full_export):
    args, actor, calls = full_export
    final = export(args)
    # Optimizer state is not an input to the exported inference weights.
    (actor / "optim_world_size_2_rank_0.pt").write_bytes(b"updated optimizer")

    assert export(args) == final
    assert len(calls) == 1


@pytest.mark.parametrize("source", ["model_world_size_2_rank_0.pt", "huggingface/config.json"])
def test_overwritten_source_is_rejected_before_latest_changes(full_export, source):
    args, actor, calls = full_export
    export(args)
    other = args.output_dir / "global_step_16"
    other.mkdir()
    update_latest_link(args.output_dir, other)
    path = actor / source
    stamp = path.stat().st_mtime_ns
    os.utime(path, ns=(stamp, stamp + 10_000_000_000))

    with pytest.raises(FileExistsError, match="source files"):
        export(args)
    assert len(calls) == 1
    assert (args.output_dir / "latest").resolve() == other


@pytest.mark.parametrize("change", ["source", "precision"])
def test_export_identity_must_match(full_export, tmp_path, change):
    args, _, calls = full_export
    export(args)
    if change == "source":
        args.checkpoint_dir = tmp_path / "another-run"
        _actor_checkpoint(args.checkpoint_dir, 8)
    else:
        args.output_dtype = "float32"

    with pytest.raises(FileExistsError, match="does not match"):
        export(args)
    assert len(calls) == 1


@pytest.mark.parametrize("changed", [False, True])
def test_legacy_export_uses_source_timestamp_check(full_export, changed):
    args, actor, calls = full_export
    final = export(args)
    marker = final / "mobilegym_export.json"
    metadata = json.loads(marker.read_text())
    del metadata["source_files"]
    marker.write_text(json.dumps(metadata))
    if changed:
        path = actor / "model_world_size_2_rank_0.pt"
        stamp = marker.stat().st_mtime_ns + 10_000_000_000
        os.utime(path, ns=(stamp, stamp))
        with pytest.raises(FileExistsError, match="source files"):
            export(args)
    else:
        assert export(args) == final
    assert len(calls) == 1


def test_source_changed_during_merge_does_not_publish(full_export, monkeypatch):
    args, actor, _ = full_export

    def merge_and_change_source(actor_dir, merged_dir):
        merged_dir.mkdir()
        (actor_dir / "model_world_size_2_rank_0.pt").write_bytes(b"new checkpoint")

    monkeypatch.setattr(
        "cookbooks.mobilegym.export_latest_checkpoint.run_verl_fsdp_merge",
        merge_and_change_source,
    )
    with pytest.raises(RuntimeError, match="source changed during"):
        export(args)
    assert not (args.output_dir / "global_step_8").exists()
    assert not (args.output_dir / "latest").is_symlink()


def test_adapter_export_tracks_base_model_changes(full_export, tmp_path, monkeypatch):
    args, _, _ = full_export
    args.checkpoint_dir = None
    args.global_step = 8
    args.lora_rank = 2
    args.adapter_dir = tmp_path / "adapter"
    args.base_model = tmp_path / "base"
    args.adapter_dir.mkdir()
    args.base_model.mkdir()
    (args.adapter_dir / "adapter_config.json").write_text("{}")
    (args.adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (args.base_model / "config.json").write_text("{}")
    weights = args.base_model / "model.safetensors"
    weights.write_bytes(b"base")

    def fake_merge(base, adapter, merged, *options):
        merged.mkdir()
        return {"merge_compute_dtype": "float32", "output_dtype": "bfloat16"}

    monkeypatch.setattr("cookbooks.mobilegym.export_latest_checkpoint.merge_lora", fake_merge)
    final = export(args)
    assert export(args) == final
    weights.write_bytes(b"replacement base")
    with pytest.raises(FileExistsError, match="source files"):
        export(args)
