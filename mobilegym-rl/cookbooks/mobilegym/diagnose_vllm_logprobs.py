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

"""Compare vLLM decode log-probabilities with vLLM prompt rescoring.

This is the first, environment-free stage of rollout/actor probability-gap
diagnostics.  It generates a fixed continuation, appends that continuation to
the prompt, and asks the same vLLM instance to rescore it through the prefill
path.  Using temperature=1 and top_p=1 makes the two reported distributions
directly comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from statistics import mean


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _prompt_lengths(value: str) -> list[int]:
    try:
        lengths = [_positive_int(part.strip()) for part in value.split(",") if part.strip()]
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not lengths:
        raise argparse.ArgumentTypeError("expected at least one comma-separated prompt length")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=_prompt_lengths, default=[64, 1024, 4096, 8192])
    parser.add_argument("--completion-length", type=_positive_int, default=32)
    parser.add_argument("--max-model-len", type=_positive_int, default=12288)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logprobs-mode", choices=("raw_logprobs", "processed_logprobs"), default="raw_logprobs")
    return parser.parse_args()


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values) if values else math.nan,
        "max": max(values, default=math.nan),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p99": _quantile(values, 0.99),
    }


def build_prompt_ids(tokenizer, target_length: int) -> list[int]:
    prefix = tokenizer.encode(
        "System: You operate a graphical mobile application carefully.\n"
        "User: Inspect the current state and state the next concise action.\n",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(
        "Observation: the screen contains a toolbar, several labels, and interactive controls. ",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode("\nAssistant:", add_special_tokens=False)
    if target_length < len(prefix) + len(suffix):
        raise ValueError(
            f"prompt length {target_length} is too short; minimum is {len(prefix) + len(suffix)}"
        )
    available = target_length - len(prefix) - len(suffix)
    repeated = (filler * math.ceil(available / len(filler)))[:available]
    prompt_ids = prefix + repeated + suffix
    assert len(prompt_ids) == target_length
    return prompt_ids


def _selected_logprobs(logprob_dicts, token_ids: list[int]) -> list[float]:
    if logprob_dicts is None or len(logprob_dicts) != len(token_ids):
        raise RuntimeError(
            f"logprob/token length mismatch: logprobs={None if logprob_dicts is None else len(logprob_dicts)}, "
            f"tokens={len(token_ids)}"
        )
    selected = []
    for position, (candidates, token_id) in enumerate(zip(logprob_dicts, token_ids, strict=True)):
        if candidates is None or token_id not in candidates:
            raise RuntimeError(f"selected token {token_id} missing from logprobs at position {position}")
        selected.append(float(candidates[token_id].logprob))
    return selected


def compare_case(llm, tokenizer, prompt_length: int, completion_length: int, seed: int):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompt_ids = build_prompt_ids(tokenizer, prompt_length)
    generation_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=completion_length,
        min_tokens=completion_length,
        ignore_eos=True,
        logprobs=0,
        seed=seed,
    )
    generated = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        generation_params,
        use_tqdm=False,
    )[0]
    completion_ids = list(generated.outputs[0].token_ids)
    decode_logprobs = _selected_logprobs(generated.outputs[0].logprobs, completion_ids)

    rescore_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        prompt_logprobs=0,
        logprobs=0,
        seed=seed,
    )
    rescored = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids + completion_ids)],
        rescore_params,
        use_tqdm=False,
    )[0]
    prompt_logprobs = rescored.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM returned no prompt_logprobs")
    prefill_logprobs = []
    for completion_position, token_id in enumerate(completion_ids):
        absolute_position = len(prompt_ids) + completion_position
        candidates = prompt_logprobs[absolute_position]
        if candidates is None or token_id not in candidates:
            raise RuntimeError(
                f"completion token {token_id} missing from prompt_logprobs at absolute position {absolute_position}"
            )
        prefill_logprobs.append(float(candidates[token_id].logprob))

    signed_logprob_diff = [decode - prefill for decode, prefill in zip(decode_logprobs, prefill_logprobs, strict=True)]
    abs_logprob_diff = [abs(value) for value in signed_logprob_diff]
    abs_prob_diff = [
        abs(math.exp(decode) - math.exp(prefill))
        for decode, prefill in zip(decode_logprobs, prefill_logprobs, strict=True)
    ]
    outlier_indices = sorted(range(len(abs_prob_diff)), key=abs_prob_diff.__getitem__, reverse=True)[:8]
    return {
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
        "completion_ids": completion_ids,
        "completion_text": tokenizer.decode(completion_ids),
        "decode_logprobs": decode_logprobs,
        "prefill_logprobs": prefill_logprobs,
        "abs_probability_difference": summarize(abs_prob_diff),
        "abs_logprob_difference": summarize(abs_logprob_diff),
        "signed_logprob_difference_mean": mean(signed_logprob_diff),
        "outliers": [
            {
                "position": index,
                "token_id": completion_ids[index],
                "token": tokenizer.decode([completion_ids[index]]),
                "decode_logprob": decode_logprobs[index],
                "prefill_logprob": prefill_logprobs[index],
                "abs_probability_difference": abs_prob_diff[index],
            }
            for index in outlier_indices
        ],
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    if max(args.prompt_lengths) + args.completion_length + 1 > args.max_model_len:
        raise ValueError("max prompt plus completion exceeds --max-model-len")

    from transformers import AutoTokenizer
    from vllm import LLM

    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        logprobs_mode=args.logprobs_mode,
    )

    results = []
    for prompt_length in args.prompt_lengths:
        print(f"[diagnose] prompt_length={prompt_length}", flush=True)
        result = compare_case(llm, tokenizer, prompt_length, args.completion_length, args.seed)
        results.append(result)
        print(
            "[diagnose] probability_diff_mean="
            f"{result['abs_probability_difference']['mean']:.8f} "
            f"max={result['abs_probability_difference']['max']:.8f}",
            flush=True,
        )

    payload = {
        "model": str(model_path),
        "vllm_version": __import__("vllm").__version__,
        "device_count": __import__("torch").cuda.device_count(),
        "settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs_mode": args.logprobs_mode,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "seed": args.seed,
        },
        "cases": results,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(f"[diagnose] wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
