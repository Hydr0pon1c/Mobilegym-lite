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

"""Weight-free Qwen3-VL training-input gate, shared by native and visual runs.

Uses the real processor, gateway trace reconstruction, rLLM batching/M-RoPE,
VERL no-padding conversion and FSDP input preparation. Does not create a model,
start Ray/vLLM, generate trajectories or perform an optimizer update.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys


def runtime_identity():
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "vllm", "ray", "tensordict", "qwen-vl-utils", "playwright"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {}
    for name in ("rllm", "verl", "transformers", "torch"):
        spec = importlib.util.find_spec(name)
        sources[name] = spec.origin if spec else None
    return {"python": sys.executable, "pythonVersion": platform.python_version(),
            "versions": versions, "sources": sources}


def check_inputs(model_path):
    print("[training-input-check] importing processor/actor input utilities (no model weights)", flush=True)
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from types import SimpleNamespace

    from rllm.experimental.engine.trace_converter import build_multimodal_inputs
    from rllm.experimental.verl.dataclass import AccumulatedData, ProcessedStepData
    from rllm.experimental.verl.transform import _batch_tensors_and_build_data_proto
    from rllm.workflows.workflow import TerminationReason
    from verl.utils import hf_processor, tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.utils.padding import left_right_2_no_padding

    processor = hf_processor(str(model_path), local_files_only=True)
    if processor is None or processor.__class__.__name__ != "Qwen3VLProcessor":
        raise ValueError("Training-input gate requires a working Qwen3VLProcessor")
    print("[training-input-check] processor ready; checking unequal prompts and packed actor inputs", flush=True)
    accumulated = AccumulatedData()
    raw_shapes, original_pixels = [], []
    for i, text in enumerate(("Read this screen.", "Read this screen. " + "A longer history. " * 64, "Text-only control.")):
        content = [{"type": "text", "text": text}]
        if i < 2:
            content.insert(0, {"type": "image", "image": Image.new("RGB", (360, 800), (120, 140, 160))})
        messages = [{"role": "user", "content": content}]
        images, videos = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        raw = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")
        visual = build_multimodal_inputs(messages, processor, disable_thinking=True) or {}
        raw_shapes.append({k: list(v.shape) for k, v in visual.items()})
        if i < 2:
            torch.testing.assert_close(visual["pixel_values"], raw["pixel_values"].to(torch.bfloat16))
            torch.testing.assert_close(visual["image_grid_thw"], raw["image_grid_thw"])
            original_pixels.append(visual["pixel_values"].clone())
        response = torch.tensor(processor.tokenizer.encode("OK.", add_special_tokens=False))
        step = ProcessedStepData(prompt=raw["input_ids"][0], response=response,
                                 mask=torch.ones(len(response)), step_reward=1.0,
                                 step_id=str(i), multi_modal_inputs=visual,
                                 logprobs=[-0.5] * len(response))
        accumulated.add_step(step, str(i), 1.0, 1, True)
        accumulated.episode_ids.append(str(i))
        accumulated.is_correct.append(True)
        accumulated.termination_reasons.append(TerminationReason.ENV_DONE)
        accumulated.metrics.append({})
        accumulated.repeat_counts.append(1)
    lengths = [len(p) for p in accumulated.prompts]
    if lengths[0] == lengths[1]:
        raise AssertionError("Gate must exercise unequal prompt lengths")
    max_prompt = max(lengths) + 3
    max_response = max(map(len, accumulated.responses)) + 2
    batch = _batch_tensors_and_build_data_proto(
        accumulated, processor.tokenizer.pad_token_id, max_prompt, max_response, processor)
    if batch.batch["position_ids"].shape != (3, 4, max_prompt + max_response):
        raise AssertionError("Qwen3-VL must retain four-axis packed position metadata")
    if any("mm_token_type_ids" in value for value in batch.non_tensor_batch["multi_modal_inputs"]):
        raise AssertionError("Prompt-only mm_token_type_ids leaked into actor multimodal collation")
    packed = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(packed, temperature=0.7, use_remove_padding=True, use_fused_kernels=True)
    # This method only prepares tensors. No engine/model/distributed group is
    # constructed; use_ulysses_sp=False matches the existing one-rank-per-actor profile.
    inputs, _ = FSDPEngineWithLMHead.prepare_model_inputs(SimpleNamespace(use_ulysses_sp=False), packed)
    torch.testing.assert_close(inputs["pixel_values"], torch.cat(original_pixels))
    torch.testing.assert_close(inputs["input_ids"].squeeze(0),
                               batch.batch["input_ids"][batch.batch["attention_mask"].bool()])
    expected_positions = torch.cat([
        batch.batch["position_ids"][i, :, batch.batch["attention_mask"][i].bool()]
        for i in range(3)
    ], dim=1).unsqueeze(1)
    torch.testing.assert_close(inputs["position_ids"], expected_positions)
    if "mm_token_type_ids" in inputs:
        raise AssertionError("Unaligned token metadata reached the actor")
    return {"processor": type(processor).__name__, "promptLengths": lengths,
            "rawVisualShapes": raw_shapes,
            "actorInputShapes": {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            "checks": ["gateway-visual-reconstruction", "unequal-prompts", "mixed-text-image",
                       "position-ids-before-cleanup", "dataproto-no-padding", "fsdp-prepare-model-inputs",
                       "pixel-identity", "token-and-position-alignment"]}


def run_check(model_path, output_path):
    model_path, output_path = Path(model_path), Path(output_path)
    report = {"passed": False, "model": str(model_path.resolve()), "runtime": runtime_identity(),
              "limitations": ["No weights, model forward/backward, optimizer, Ray or vLLM execution"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"[training-input-check] runtime: {json.dumps(report['runtime'])}", flush=True)
    try:
        local_rl = Path(__file__).resolve().parents[2]
        local_verl = Path(os.environ.get("LOCAL_VERL_ROOT", local_rl / "verl"))
        for name, root in (("rllm", local_rl / "rllm"), ("verl", local_verl / "verl")):
            source = report["runtime"]["sources"][name]
            if not source or not Path(source).resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Unexpected {name} import source: {source}; expected {root}")
        report.update(check_inputs(model_path))
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    print(f"[training-input-check] passed; report: {output_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_check(args.model, args.output)


if __name__ == "__main__":
    main()
