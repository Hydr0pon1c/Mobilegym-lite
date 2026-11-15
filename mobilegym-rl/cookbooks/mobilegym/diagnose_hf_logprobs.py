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

"""Rescore fixed vLLM continuations with the Transformers actor path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch

from diagnose_vllm_logprobs import atomic_write_json, build_prompt_ids, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def _model_class(model_path: Path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = config.architectures or []
    if architectures and "ForConditionalGeneration" in architectures[0]:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


@torch.inference_mode()
def rescore_case(model, tokenizer, case: dict, device: str) -> dict:
    prompt_ids = build_prompt_ids(tokenizer, int(case["prompt_length"]))
    completion_ids = [int(token_id) for token_id in case["completion_ids"]]
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=len(completion_ids) + 1,
        return_dict=True,
    )
    logits = outputs.logits[0]
    if logits.shape[0] != len(completion_ids) + 1:
        raise RuntimeError(
            f"unexpected logits length {logits.shape[0]}, expected {len(completion_ids) + 1}"
        )
    prediction_logits = logits[:-1].float()
    labels = torch.tensor(completion_ids, dtype=torch.long, device=prediction_logits.device)
    actor_logprobs = (
        prediction_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        - torch.logsumexp(prediction_logits, dim=-1)
    ).cpu().tolist()

    vllm_prefill_logprobs = [float(value) for value in case["prefill_logprobs"]]
    vllm_decode_logprobs = [float(value) for value in case["decode_logprobs"]]
    signed_prefill_logprob_diff = [
        actor - rollout
        for actor, rollout in zip(actor_logprobs, vllm_prefill_logprobs, strict=True)
    ]
    abs_prefill_logprob_diff = [abs(value) for value in signed_prefill_logprob_diff]
    abs_prefill_prob_diff = [
        abs(math.exp(actor) - math.exp(rollout))
        for actor, rollout in zip(actor_logprobs, vllm_prefill_logprobs, strict=True)
    ]
    abs_decode_prob_diff = [
        abs(math.exp(actor) - math.exp(rollout))
        for actor, rollout in zip(actor_logprobs, vllm_decode_logprobs, strict=True)
    ]
    outlier_indices = sorted(
        range(len(abs_decode_prob_diff)), key=abs_decode_prob_diff.__getitem__, reverse=True
    )[:8]
    return {
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
        "actor_logprobs": actor_logprobs,
        "actor_vs_vllm_prefill_probability_difference": summarize(abs_prefill_prob_diff),
        "actor_vs_vllm_prefill_logprob_difference": summarize(abs_prefill_logprob_diff),
        "actor_minus_vllm_prefill_logprob_mean": mean(signed_prefill_logprob_diff),
        "actor_vs_vllm_decode_probability_difference": summarize(abs_decode_prob_diff),
        "outliers": [
            {
                "position": index,
                "token_id": completion_ids[index],
                "token": tokenizer.decode([completion_ids[index]]),
                "actor_logprob": actor_logprobs[index],
                "vllm_decode_logprob": vllm_decode_logprobs[index],
                "vllm_prefill_logprob": vllm_prefill_logprobs[index],
                "actor_vs_vllm_decode_abs_probability_difference": abs_decode_prob_diff[index],
            }
            for index in outlier_indices
        ],
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.vllm_results.read_text(encoding="utf-8"))
    if any("completion_ids" not in case for case in source["cases"]):
        raise ValueError("vLLM result lacks completion token data; rerun the updated diagnostic")

    from transformers import AutoTokenizer

    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = _model_class(model_path).from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.to(args.device)
    model.eval()

    results = []
    for source_case in source["cases"]:
        print(f"[diagnose] actor prompt_length={source_case['prompt_length']}", flush=True)
        result = rescore_case(model, tokenizer, source_case, args.device)
        results.append(result)
        summary = result["actor_vs_vllm_decode_probability_difference"]
        print(
            f"[diagnose] actor_vs_decode mean={summary['mean']:.8f} max={summary['max']:.8f}",
            flush=True,
        )

    payload = {
        "model": str(model_path),
        "vllm_results": str(args.vllm_results.resolve()),
        "attn_implementation": args.attn_implementation,
        "dtype": "bfloat16",
        "cases": results,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(f"[diagnose] wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
