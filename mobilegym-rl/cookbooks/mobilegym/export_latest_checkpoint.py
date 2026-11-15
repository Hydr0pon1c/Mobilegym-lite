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

"""Export a VERL FSDP checkpoint or PEFT adapter as a standalone HF model.

For LoRA checkpoints, the VERL merger reconstructs a base model and a PEFT
adapter.  This wrapper supplies metadata omitted by PPO checkpoints, then
folds the adapter into the base weights.  Full-parameter checkpoints are
reconstructed directly.  Both paths produce a directory loadable by vLLM.

LoRA folding defaults to FP32 and the completed model is cast to BF16 exactly
once, avoiding the extra rounding incurred by folding directly into BF16.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPORT_MARKER = "mobilegym_export.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a VERL FSDP checkpoint or adapter into a vLLM-ready Hugging Face model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", type=Path, help="Run-level VERL checkpoint directory")
    source.add_argument("--adapter-dir", type=Path, help="Existing PEFT adapter directory")
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Base Hugging Face model (required with --adapter-dir; reconstructed for checkpoints)",
    )
    parser.add_argument(
        "--global-step",
        type=_positive_int,
        help="Exact checkpoint step, or the adapter's step (required with --adapter-dir)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Run-level directory for exported models")
    parser.add_argument(
        "--lora-rank",
        type=_non_negative_int,
        default=0,
        help="LoRA rank for a legacy adapter checkpoint (default: 0/full parameter)",
    )
    parser.add_argument("--lora-alpha", type=_positive_int, default=1)
    parser.add_argument("--lora-exclude-modules", default=r".*visual.*")
    parser.add_argument("--device", default="cpu", help="Device used for PEFT merge, e.g. cpu or cuda:0")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--merge-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Arithmetic dtype used while folding LoRA weights (default: float32)",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="Stored floating-point dtype after the merge (default: bfloat16)",
    )
    return parser.parse_args()


def resolve_actor(checkpoint_dir: Path, requested_step: int | None = None) -> tuple[int, Path]:
    checkpoint_dir = checkpoint_dir.resolve()
    if requested_step is not None:
        step = requested_step
    else:
        marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
        if marker.is_file():
            try:
                step = int(marker.read_text(encoding="utf-8").strip())
            except ValueError as exc:
                raise ValueError(f"Invalid checkpoint marker: {marker}") from exc
        else:
            steps = []
            for path in checkpoint_dir.glob("global_step_*/actor"):
                try:
                    steps.append(int(path.parent.name.removeprefix("global_step_")))
                except ValueError:
                    continue
            if not steps:
                raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
            step = max(steps)

    if step <= 0:
        raise ValueError(f"Checkpoint step must be positive, got {step}")
    actor_dir = checkpoint_dir / f"global_step_{step}" / "actor"
    required = (actor_dir / "fsdp_config.json", actor_dir / "huggingface" / "config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete actor checkpoint; missing: {', '.join(missing)}")
    if not any(actor_dir.glob("model_world_size_*_rank_0.pt")):
        raise FileNotFoundError(f"No rank-0 model shard found in {actor_dir}")
    return step, actor_dir


def resolve_latest_actor(checkpoint_dir: Path) -> tuple[int, Path]:
    """Backward-compatible helper used by callers that want the latest step."""

    return resolve_actor(checkpoint_dir)


def ensure_lora_metadata(actor_dir: Path, rank: int, alpha: int, exclude_modules: str) -> None:
    metadata_path = actor_dir / "lora_train_meta.json"
    expected = {
        "r": rank,
        "lora_alpha": alpha,
        "task_type": "CAUSAL_LM",
        "exclude_modules": exclude_modules,
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("r", "lora_alpha", "task_type"):
            if existing.get(key) != expected[key]:
                raise ValueError(
                    f"LoRA metadata mismatch in {metadata_path}: "
                    f"{key}={existing.get(key)!r}, expected {expected[key]!r}"
                )
        if existing.get("exclude_modules") == exclude_modules:
            return

    tmp_path = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, metadata_path)


def run_verl_fsdp_merge(actor_dir: Path, stage_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(stage_dir),
        "--trust-remote-code",
    ]
    print("[export] Reconstructing FSDP shards with VERL", flush=True)
    subprocess.run(command, check=True)


def patch_adapter_config(adapter_dir: Path, rank: int, alpha: int, exclude_modules: str) -> None:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"VERL merger did not produce a LoRA adapter: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["r"] = rank
    config["lora_alpha"] = alpha
    config["task_type"] = "CAUSAL_LM"
    config["exclude_modules"] = exclude_modules
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _model_class(model_dir: Path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    architectures = config.architectures or []
    if architectures and "ForConditionalGeneration" in architectures[0]:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def merge_lora(
    base_model_dir: Path,
    adapter_dir: Path,
    merged_dir: Path,
    device: str,
    max_shard_size: str,
    merge_dtype_name: str,
    output_dtype_name: str,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, AutoTokenizer

    merge_dtype = _torch_dtype(merge_dtype_name)
    output_dtype = _torch_dtype(output_dtype_name)
    model_class = _model_class(base_model_dir)
    load_kwargs = {
        "dtype": merge_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if device != "cpu":
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested export device {device!r}, but torch.cuda is unavailable")
        load_kwargs["device_map"] = {"": device}

    print(f"[export] Loading base model on {device} in {merge_dtype_name}", flush=True)
    base_model = model_class.from_pretrained(base_model_dir, **load_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    peft_model.to(dtype=merge_dtype)
    adapter_dtypes = Counter(
        str(parameter.dtype)
        for name, parameter in peft_model.named_parameters()
        if "lora_" in name and parameter.is_floating_point()
    )
    if set(adapter_dtypes) != {str(merge_dtype)}:
        raise ValueError(f"LoRA tensors were not normalized to {merge_dtype}: {dict(adapter_dtypes)}")

    print(f"[export] Folding LoRA weights in {merge_dtype_name}", flush=True)
    merged_model = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
    merged_dtypes = Counter(
        str(parameter.dtype) for parameter in merged_model.parameters() if parameter.is_floating_point()
    )
    if set(merged_dtypes) != {str(merge_dtype)}:
        raise ValueError(f"Merged model has unexpected compute dtypes: {dict(merged_dtypes)}")

    if output_dtype != merge_dtype:
        print(f"[export] Casting merged model once to {output_dtype_name}", flush=True)
        merged_model.to(dtype=output_dtype)
    merged_model.config.dtype = output_dtype

    merged_dir.mkdir(parents=True)
    merged_model.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True, use_fast=True)
    processor.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    del merged_model
    del peft_model
    del base_model

    return {
        "merge_compute_dtype": merge_dtype_name,
        "output_dtype": output_dtype_name,
        "adapter_compute_dtypes": dict(adapter_dtypes),
        "pre_cast_merged_dtypes": dict(merged_dtypes),
    }


def validate_export(model_dir: Path, expected_dtype: str) -> dict[str, object]:
    from safetensors import safe_open

    # Validation needs stored metadata, not dynamic model classes. Importing
    # Transformers here can scan every installed distribution over the shared
    # filesystem, even after the reconstructed weights were written successfully.
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")

    total_keys = 0
    bad_keys = []
    dtype_counts: Counter[str] = Counter()
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                dtype_counts[handle.get_slice(key).get_dtype()] += 1
        total_keys += len(keys)
        bad_keys.extend(key for key in keys if "lora_" in key or ".base_layer." in key)
    if bad_keys:
        raise ValueError(f"Export still contains PEFT keys: {bad_keys[:5]}")
    if total_keys == 0:
        raise ValueError(f"Export contains no tensor keys: {model_dir}")
    if (model_dir / "adapter_config.json").exists() or (model_dir / "lora_adapter").exists():
        raise ValueError(f"Export is still an adapter model: {model_dir}")

    expected_safetensors_dtype = {"float32": "F32", "bfloat16": "BF16"}[expected_dtype]
    unexpected_float_dtypes = {
        dtype: count
        for dtype, count in dtype_counts.items()
        if dtype in {"F16", "BF16", "F32", "F64"} and dtype != expected_safetensors_dtype
    }
    if unexpected_float_dtypes:
        raise ValueError(
            f"Export contains floating tensors outside {expected_safetensors_dtype}: {unexpected_float_dtypes}"
        )

    dtype = str(config.get("dtype", config.get("torch_dtype", "unknown")))
    if dtype in {"float16", "bfloat16", "float32", "float64"}:
        dtype = f"torch.{dtype}"
    return {
        "architecture": (config.get("architectures") or [None])[0],
        "dtype": dtype,
        "weight_files": len(weight_files),
        "tensor_keys": total_keys,
        "weight_dtype_counts": dict(dtype_counts),
    }


def update_latest_link(output_dir: Path, final_dir: Path) -> None:
    latest = output_dir / "latest"
    if latest.exists() and not latest.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink path: {latest}")
    tmp_link = output_dir / f".latest.{os.getpid()}.tmp"
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(final_dir.name)
    os.replace(tmp_link, latest)


def source_snapshot(
    actor_dir: Path | None, adapter_dir: Path | None, base_dir: Path | None
) -> dict[str, dict[str, int]]:
    """Record export inputs without reading multi-gigabyte weights or optimizer state."""
    if actor_dir is not None:
        files = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
        files.append(actor_dir / "fsdp_config.json")
        files.extend((actor_dir / "huggingface").rglob("*"))
        if (actor_dir / "lora_train_meta.json").is_file():
            files.append(actor_dir / "lora_train_meta.json")
    else:
        files = [path for root in (adapter_dir, base_dir) if root is not None for path in root.rglob("*")]
    result = {}
    for path in sorted(set(files)):
        if path.is_file():
            stat = path.stat()
            result[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def check_existing_export(marker_path: Path, expected: dict, snapshot: dict) -> None:
    metadata = json.loads(marker_path.read_text(encoding="utf-8"))
    changed = [key for key, value in expected.items() if metadata.get(key) != value]
    if "source_files" in metadata:
        source_changed = metadata["source_files"] != snapshot
    else:
        # Legacy exports have no snapshot. Detect sources rewritten after export;
        # this timestamp check is not a content checksum.
        exported_at = datetime.fromisoformat(metadata["exported_at"])
        export_ns = int(exported_at.timestamp() * 1_000_000_000)
        source_changed = any(info["mtime_ns"] > export_ns for info in snapshot.values())
    if changed or source_changed:
        reason = ", ".join(changed + (["source files"] if source_changed else []))
        raise FileExistsError(
            f"Existing export does not match the requested checkpoint/options ({reason}): "
            f"{marker_path.parent}. Use a new --output-dir, or remove this exported model "
            "directory before exporting again."
        )


def export(args: argparse.Namespace) -> Path:
    is_lora = args.lora_rank > 0
    if args.adapter_dir is not None:
        if not is_lora:
            raise ValueError("--adapter-dir requires --lora-rank greater than 0")
        if args.base_model is None:
            raise ValueError("--base-model is required with --adapter-dir")
        if args.global_step is None:
            raise ValueError("--global-step is required with --adapter-dir")
        step = args.global_step
        actor_dir = None
        base_model_dir = args.base_model.resolve()
        source_adapter_dir = args.adapter_dir.resolve()
        required = (
            base_model_dir / "config.json",
            source_adapter_dir / "adapter_config.json",
            source_adapter_dir / "adapter_model.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete adapter export source; missing: {', '.join(missing)}")
    else:
        if args.base_model is not None:
            raise ValueError("--base-model is only valid with --adapter-dir")
        step, actor_dir = resolve_actor(args.checkpoint_dir, args.global_step)
        base_model_dir = None
        source_adapter_dir = None

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / f"global_step_{step}"
    marker_path = final_dir / EXPORT_MARKER
    expected = {
        "source_checkpoint": str(actor_dir) if actor_dir is not None else None,
        "source_adapter": str(source_adapter_dir) if source_adapter_dir is not None else None,
        "base_model": str(args.base_model.resolve()) if args.base_model is not None else None,
        "global_step": step,
        "training_mode": "lora" if is_lora else "full",
        "output_dtype": args.output_dtype,
        "merge_compute_dtype": args.merge_dtype if is_lora else "checkpoint",
    }
    if is_lora:
        expected.update(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_exclude_modules=args.lora_exclude_modules,
        )
    if marker_path.is_file():
        check_existing_export(
            marker_path, expected, source_snapshot(actor_dir, source_adapter_dir, base_model_dir)
        )
        print(f"[export] Existing completed export found: {final_dir}", flush=True)
        update_latest_link(output_dir, final_dir)
        return final_dir
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete export: {final_dir}")

    if actor_dir is not None and is_lora:
        ensure_lora_metadata(actor_dir, args.lora_rank, args.lora_alpha, args.lora_exclude_modules)
    snapshot = source_snapshot(actor_dir, source_adapter_dir, base_model_dir)
    with tempfile.TemporaryDirectory(prefix=f".global_step_{step}.", dir=output_dir) as work_dir_str:
        work_dir = Path(work_dir_str)
        merged_dir = work_dir / "merged"
        if is_lora:
            if actor_dir is not None:
                stage_dir = work_dir / "unmerged"
                run_verl_fsdp_merge(actor_dir, stage_dir)
                adapter_dir = stage_dir / "lora_adapter"
                base_model_dir = stage_dir
            else:
                adapter_dir = work_dir / "lora_adapter"
                shutil.copytree(source_adapter_dir, adapter_dir)

            patch_adapter_config(adapter_dir, args.lora_rank, args.lora_alpha, args.lora_exclude_modules)
            precision = merge_lora(
                base_model_dir,
                adapter_dir,
                merged_dir,
                args.device,
                args.max_shard_size,
                args.merge_dtype,
                args.output_dtype,
            )
        else:
            assert actor_dir is not None
            print("[export] Reconstructing full-parameter actor", flush=True)
            run_verl_fsdp_merge(actor_dir, merged_dir)
            precision = {
                "merge_compute_dtype": "checkpoint",
                "output_dtype": args.output_dtype,
            }
        validation = validate_export(merged_dir, args.output_dtype)
        # In the checkpoint/LoRA path base_model_dir now points at the temporary
        # reconstruction, while the input remains the original actor checkpoint.
        current_source = source_snapshot(
            actor_dir, source_adapter_dir, args.base_model.resolve() if args.base_model else None
        )
        if current_source != snapshot:
            raise RuntimeError(
                "Export source changed during the merge; no model was published. "
                "Retry after checkpoint writing finishes."
            )
        export_metadata = {
            **expected,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_exclude_modules": args.lora_exclude_modules,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_files": snapshot,
            **precision,
            **validation,
        }
        (merged_dir / EXPORT_MARKER).write_text(
            json.dumps(export_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(merged_dir, final_dir)

    update_latest_link(output_dir, final_dir)
    print(f"[export] vLLM-ready model: {final_dir}", flush=True)
    return final_dir


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
