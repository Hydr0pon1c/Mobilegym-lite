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

"""Model-free setup audit and optional GUI replay for resource, feature or Lite setup.

Run ``python -m bench_env.completeness_check --help``. Evidence lives in the
specified output directory, never in source control. Replay uses fresh visible
targets from the retained reference recipes, then calls the native verifier.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zlib

from bench_env.completeness import LITE_VERSION, MODES, TASK_IDS, VERSION, apply_completeness, query_answer
from bench_env import feature_completeness as features
from bench_env.env.base import Action, ActionType
from bench_env.env.pool import EnvPool
from bench_env.runner.base import Controller, Evaluator
from bench_env.task.registry import TaskRegistry
from bench_env.splits import resolve_split


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def make_task(task_id: str, sample_seed: int):
    seed = (sample_seed ^ zlib.crc32(f"{task_id}:0".encode())) & 0xffffffff
    return TaskRegistry().create_task(task_id, _seed=seed)


def mode_rollouts(modes):
    for mode in modes:
        for index in range(features.MIX_PERIOD if mode == features.MIX_MODE else 1):
            yield mode, index


async def scan(env, task_id: str, seed: int, out: Path, modes=MODES) -> dict:
    task = make_task(task_id, seed)
    original, params = await Controller.setup(env, task, eval_mode="text")
    description = task.description
    evaluator = Evaluator(judge_mode="state", eval_mode="text")
    execution = SimpleNamespace(agent_answer=None)
    initial_judge = evaluator._evaluate_with_state(task, original, original, execution).to_dict()
    profiles = []
    for mode, index in mode_rollouts(modes):
        obs = await apply_completeness(env, task, original, mode, rollout_index=index)
        if mode in features.MODES and obs.state["apps"] != original.state["apps"]:
            from bench_env.reference import changed_paths
            write(out / "states" / f"{task_id}--{seed}--{mode}--{index}-mismatch.json", {
                "before": original.state["apps"], "after": obs.state["apps"],
                "changed_paths": changed_paths(original.state["apps"], obs.state["apps"]),
            })
            raise ValueError("Feature mask changed app data/settings")
        if mode in features.MODES:
            if obs.state.get("os", {}).get("providers") != original.state.get("os", {}).get("providers"):
                raise ValueError("Feature mask changed provider data")
            current_judge = evaluator._evaluate_with_state(task, original, obs, execution).to_dict()
            if current_judge != initial_judge or current_judge.get("judge_error"):
                raise ValueError(f"Feature mask changed or errored the initial native judge: {current_judge}")
        if params != task.params or description != task.description:
            raise ValueError("Projection changed task parameters/instruction")
        if query_answer(task, obs) != query_answer(task, original):
            raise ValueError("Query answer changed")
        profiles.append(deepcopy(task._completeness_record))
    # Save prepared input so rules can be reviewed and tested without a browser.
    write(out / "states" / f"{task_id}--{seed}.json", {
        "task_id": task_id, "sample_seed": seed, "params": params,
        "description": description, "state": original.state,
    })
    return {"task_id": task_id, "sample_seed": seed, "profiles": profiles,
            "initial_judge": initial_judge,
            "effective": any(p["effective"] for p in profiles), "background_only": profiles[-1].get("background_only", False),
            "query_answer_available": query_answer(task, original)[0]}


def target_matches(expected: dict, control: dict) -> bool:
    actual = control.get("identity") or {}
    if expected["kind"] != actual.get("kind") or control.get("disabled"):
        return False
    if expected["kind"] in ("trigger", "action"):
        return (expected["id"] == actual.get("id")
                and expected.get("params", {}) == actual.get("params", {})
                and (not expected.get("row_label") or expected["row_label"] == control.get("label")))
    if expected["kind"] == "label":
        return expected["label"] == actual.get("label") and expected.get("tag") == actual.get("tag")
    if expected["kind"] == "context":
        return (expected.get("context") == actual.get("context")
                and expected.get("tag") == actual.get("tag")
                and (not expected.get("icon") or expected["icon"] == actual.get("icon")))
    if expected["kind"] == "field":
        return all(expected.get(key) == actual.get(key) for key in ("name", "tag", "type"))
    return False


def matching_controls(target: dict, controls: list[dict]) -> list[dict]:
    matches = [c for c in controls if target_matches(target, c)]
    if len(matches) > 1 and target.get("context"):
        matches = [c for c in matches if (c.get("identity") or {}).get("context") == target["context"]]
    return matches


async def replay(env, task_id: str, mode: str, index: int, bundle: dict, out: Path) -> dict:
    from bench_env.reference import DOM_PROBE
    task = make_task(task_id, 41)
    initial, _ = await Controller.setup(env, task, eval_mode="text", completeness=mode,
                                      completeness_rollout_index=index)
    entry = bundle["by_task"][task_id]
    if task.description != entry["task_name"]:
        raise ValueError("Seed-41 instruction differs from reference")
    suffix = f"--{index}" if mode == features.MIX_MODE else ""
    folder = out / "replays" / f"{task_id}--{mode}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    await env.page.screenshot(path=str(folder / "initial.jpg"), type="jpeg", quality=80)
    steps, answer = [], None
    for operation in entry["operations"]:
        data = deepcopy(operation["_motor"])
        if operation["action"] in ("CLICK", "LONG_PRESS"):
            target = operation.get("target")
            if not target:
                raise ValueError("Recipe lacks a semantic click target")
            dom = await env.page.evaluate(DOM_PROBE, {"point": None, "gesture": operation["action"], "probes": False})
            matches = matching_controls(target, dom["controls"])
            if len(matches) != 1:
                write(folder / "unresolved.json", {"target": target, "dom": dom})
                raise ValueError(f"Expected one visible target, got {len(matches)}: {target}")
            rect = matches[0]["rect"]
            width, height = dom["viewport"]
            data["point"] = [(rect["x"] + rect["width"] / 2) * 1000 / width,
                             (rect["y"] + rect["height"] / 2) * 1000 / height]
        action = Action(ActionType(operation["action"]), data)
        await env.step(action)
        steps.append({"action": operation["action"], "data": data, "route": await env.get_route()})
        if action.action_type == ActionType.ANSWER:
            answer = data["value"]
    last = await env.get_observation()
    judge = Evaluator(judge_mode="state", eval_mode="text")._evaluate_with_state(
        task, initial, last, SimpleNamespace(agent_answer=answer))
    record = {"task_id": task_id, "mode": mode, "rollout_index": index, "judge": judge.to_dict(),
              "completeness": task._completeness_record, "steps": steps}
    write(folder / "result.json", record)
    await env.page.screenshot(path=str(folder / "final.jpg"), type="jpeg", quality=80)
    return record


async def main(args) -> None:
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        raise ValueError("Output already contains an audit; choose a new output directory")
    (out / "states").mkdir(exist_ok=True)
    modes, task_ids, version = {
        "resources": (MODES, TASK_IDS, VERSION),
        "features": (features.STATIC_MODES, features.TASK_IDS, features.VERSION),
        "feature-mix": ((features.MIX_MODE,), features.TASK_IDS, features.MIX_MODE),
        "feature-layout": (("feature0", features.LAYOUT_MODE), features.TASK_IDS, features.LAYOUT_VERSION),
        "feature-selective": (("feature0", features.SELECTIVE_MODE), features.TASK_IDS, features.SELECTIVE_MODE),
        "feature-plus": ((features.SELECTIVE_MODE, features.PLUS_MODE), features.PLUS_TASK_IDS, features.PLUS_MODE),
        "lite": ((LITE_VERSION,), TASK_IDS | features.TASK_IDS, LITE_VERSION),
    }[args.axis]
    # Explicit splits also audit full fallbacks, so coverage uses the whole
    # training set as its denominator rather than only the supported templates.
    ids = sorted(resolve_split(args.split) if args.split else set(args.task_id or task_ids))
    if not ids:
        raise ValueError("No tasks selected")
    registry = TaskRegistry()
    for tid in ids:
        registry.get_by_id(tid)
    bundle, url = None, args.env_url
    if args.reference_bundle:
        from bench_env.reference_runtime import load_reference_bundle, reference_url
        bundle = load_reference_bundle(args.reference_bundle)
        url = reference_url(url, bundle)
    if args.replay_task_id and bundle is None:
        raise ValueError("GUI recipes require --reference-bundle")
    if set(args.replay_task_id) - set(ids):
        raise ValueError("Replay tasks must be in the audited task set")
    root = Path(__file__).resolve().parents[1]
    sources = (Path(__file__), Path(__file__).with_name("completeness.py"), Path(features.__file__))
    (out / "source").mkdir(exist_ok=True)
    for source in sources:
        (out / "source" / source.name).write_bytes(source.read_bytes())
    write(out / "meta.json", {"version": version, "args": vars(args), "env_url": url, "tasks": ids,
          "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
          "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sources}})
    results = {"setup": [], "replay": []}
    async with EnvPool(url, n=args.parallel, num_browsers=args.browsers,
                       delay_after_action=.8, verbose=False) as pool:
        for phase, jobs in (
            ("setup", [(tid, seed) for tid in ids for seed in args.sample_seeds]),
            ("replay", [(tid, mode, index) for tid in args.replay_task_id for mode, index in mode_rollouts(modes)]),
        ):
            queue = asyncio.Queue()
            for job in jobs:
                queue.put_nowait(job)
            with (out / f"{phase}.jsonl").open("w") as stream:
                async def worker(env):
                    while not queue.empty():
                        job = queue.get_nowait()
                        try:
                            coro = scan(env, *job, out, modes) if phase == "setup" else replay(env, *job, bundle, out)
                            record = await asyncio.wait_for(coro, timeout=180)
                            record["ok"] = True
                        except Exception as error:
                            record = {"task_id": job[0], "sample_seed" if phase == "setup" else "mode": job[1],
                                      **({"rollout_index": job[2]} if phase == "replay" else {}),
                                      "ok": False, "error": f"{type(error).__name__}: {error}"}
                        results[phase].append(record)
                        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                        stream.flush()
                        print(json.dumps({"phase": phase, "job": job, "ok": record["ok"],
                              "passed": record.get("judge", {}).get("passed"), "error": record.get("error")},
                              ensure_ascii=False), flush=True)
                await asyncio.gather(*(worker(env) for env in pool))
    # A template qualifies if every requested seed passed and at least one mode
    # changed resources or feature availability. Per-mode coverage is separate.
    effective = sorted(tid for tid in ids if all(
        any(r["task_id"] == tid and r["sample_seed"] == seed and r["ok"] and r.get("effective")
            for r in results["setup"]) for seed in args.sample_seeds))
    summary = {"candidate_templates": len(ids), "prepared_instances": len(results["setup"]),
               "setup_errors": sum(not r["ok"] for r in results["setup"]),
               "effective_templates": len(effective), "effective_task_ids": effective,
               "background_only_templates": sum(all(r.get("background_only") for r in results["setup"]
                   if r["task_id"] == tid) for tid in effective),
               "replays": len(results["replay"]),
               "native_passed_replays": sum(r.get("judge", {}).get("passed", False) for r in results["replay"]),
               "limitations": "State coverage is not policy exposure or learning benefit. Selected GUI recipes only."}
    summary["effective_by_mode"] = {mode: sum(
        all(any(r["task_id"] == tid and r.get("sample_seed") == seed and r["ok"] and
            any(p["mode"] == mode and p["effective"] for p in r.get("profiles", []))
            for r in results["setup"]) for seed in args.sample_seeds) for tid in ids)
        for mode in modes}
    if args.axis == "feature-mix":
        profiles = [p for r in results["setup"] if r["ok"] for p in r["profiles"]]
        summary["feature_mix"] = {
            "rollouts": len(profiles),
            "pruned": sum(p["effective"] for p in profiles),
            "full_supported": sum(p["supported"] and not p["effective"] for p in profiles),
            "full_unsupported": sum(not p["supported"] for p in profiles),
            "disabled_module_assignments": sum(len(p["disabled"]) for p in profiles),
        }
    if args.axis == "feature-layout":
        profiles = [p for r in results["setup"] if r["ok"] for p in r["profiles"]
                    if p["mode"] == features.LAYOUT_MODE]
        summary["layout"] = {
            "mode": features.LAYOUT_MODE,
            "layout_effective_instances": sum(p["layout"]["effective"] for p in profiles),
            "mask_effective_instances": sum(p["effective"] for p in profiles),
            "disabled_module_assignments": sum(len(p["disabled"]) for p in profiles),
        }
    if args.axis == "lite":
        profiles = [r["profiles"][0] for r in results["setup"] if r["ok"]]
        collections = [c for p in profiles for c in p["resources"]["collections"]]
        summary["lite"] = {
            "resource_effective_instances": sum(p["resources"]["effective"] for p in profiles),
            "feature_effective_instances": sum(p["features"]["effective"] for p in profiles),
            "both_effective_instances": sum(p["resources"]["effective"] and p["features"]["effective"]
                                             for p in profiles),
            "unchanged_instances": sum(not p["effective"] for p in profiles),
            "unsupported_instances": sum(not p["supported"] for p in profiles),
            "collection_rows_before": sum(c["before"] for c in collections),
            "collection_rows_after": sum(c["after"] for c in collections),
            "disabled_module_assignments": sum(len(p["features"]["disabled"]) for p in profiles),
            "cost_scope": "Rows in audited collections and episode-local module assignments; not bundle size or source code savings.",
        }
    write(out / "summary.json", summary)
    (out / "effective_tasks.txt").write_text("\n".join(effective) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["setup_errors"] or summary["native_passed_replays"] != summary["replays"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=["resources", "features", "feature-mix", "feature-layout", "feature-selective", "feature-plus", "lite"], default="resources")
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--output", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-id", action="append")
    selection.add_argument("--split", help="Named split or task-list path; includes unsupported/full fallbacks")
    parser.add_argument("--sample-seeds", nargs="+", type=int, default=[41, 42])
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--browsers", type=int, default=4)
    parser.add_argument("--reference-bundle")
    parser.add_argument("--replay-task-id", action="append", default=[])
    asyncio.run(main(parser.parse_args()))
