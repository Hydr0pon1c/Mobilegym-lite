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

"""Shared visual policy/lifecycle bridge. Rendering stays in window.__VISUAL__.

No model-visible metadata, task mutations, palette or permutation algorithms live here.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def load_visual_policy(value: Any) -> dict | None:
    """Load once on the driver; workers/reruns receive embedded JSON, not a path.

    Accept a canonical config, a lab snapshot, or a versioned scheduling envelope.
    Browser validation remains authoritative for capabilities and rendering axes.
    """
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Visual policy must be a JSON object")
    if "policyVersion" in value:
        if set(value) != {"policyVersion", "schedule", "config"}:
            raise ValueError("Visual policy requires exactly policyVersion, schedule, config")
        if type(value["policyVersion"]) is not int or value["policyVersion"] != 1:
            raise ValueError("Unsupported visual policyVersion")
        policy = copy.deepcopy(value)
    else:
        config = value["config"] if "engineVersion" in value else value
        policy = {"policyVersion": 1, "schedule": "fixed", "config": copy.deepcopy(config)}
    if policy["schedule"] not in ("fixed", "group_hash"):
        raise ValueError("Visual schedule must be fixed or group_hash")
    config = policy["config"]
    if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] not in (1, 2):
        raise ValueError("Visual config must use schema version 1 or 2")
    for key in ("seed", "sample"):
        if type(config.get(key)) is not int or not 0 <= config[key] <= 0xFFFFFFFF:
            raise ValueError(f"Visual {key} must be a uint32")
    _json(policy)  # Reject non-JSON values / NaN before dispatching workers.
    return policy


def validate_visual_target(policy: dict | None, url: str | None, device: str = "sim") -> None:
    if policy is None:
        return
    if device != "sim":
        raise ValueError("Visual policies require the simulator, not a real device")
    for key, value in parse_qsl(urlsplit(url or "").query, keep_blank_values=True):
        if key == "visual" or key.startswith("visual_"):
            raise ValueError("Use either a runner visual policy or visual URL parameters, not both")
        if key == "skin" and value not in ("", "default"):
            raise ValueError("Visual policies cannot be combined with legacy skins")


def require_native_val_visual(policy: Any) -> None:
    """Opt-in research guard; generic eval still supports explicit interventions."""
    policy = load_visual_policy(policy)
    if (policy is None or policy["schedule"] != "fixed"
            or policy["config"].get("theme", {}).get("strength") != 0
            or policy["config"].get("layout", {}).get("strength") != "native"):
        raise ValueError("This experiment requires an explicit fixed native validation policy")


def resolve_visual_policy(policy: dict, context: dict) -> dict:
    """Schedule by stable task/group identity; never by worker, trial or session UID.

    Eval/val identity deliberately ignores checkpoint/epoch for paired comparisons.
    Duplicate task instances within a train step share a condition (conservative).
    Hashing is deterministic, not a guarantee of finite-batch condition balance.
    """
    policy = load_visual_policy(policy)
    identity = {key: context.get(key) for key in ("mode", "task_id", "task_seed")}
    if identity["mode"] == "train":
        identity.update({key: context.get(key) for key in ("epoch", "training_step")})
        if context.get("training_group_id") is not None:
            identity["training_group_id"] = context["training_group_id"]
    if policy["schedule"] == "group_hash":
        if identity["mode"] not in ("train", "eval", "val") or not isinstance(identity["task_id"], str) or not identity["task_id"]:
            raise ValueError("group_hash needs mode and a stable task_id")
        if type(identity["task_seed"]) is not int:
            raise ValueError("group_hash needs an explicit task seed")
        if identity["mode"] == "train" and any(type(identity[key]) is not int or identity[key] < 0 for key in ("epoch", "training_step")):
            raise ValueError("Training group_hash needs nonnegative epoch and training_step metadata")
    digest = hashlib.sha256(_json(identity).encode()).hexdigest()
    config = copy.deepcopy(policy["config"])
    if policy["schedule"] == "group_hash":
        config["sample"] = (config["sample"] + int(digest[:8], 16)) & 0xFFFFFFFF
    return {
        "bridgeVersion": "visual-bridge-v1",
        "policy": policy,
        "policyId": hashlib.sha256(_json(policy).encode()).hexdigest(),
        "groupIdentity": identity,
        "groupId": digest,
        "requestedConfig": config,
    }


@lru_cache(maxsize=1)
def visual_source_revision() -> dict:
    """Runner checkout provenance, not a claim about a separately served frontend."""
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5).strip()
        # Metadata readers must not contend on the shared index. A timed-out
        # status refresh could otherwise leave index.lock behind on this mount.
        dirty = bool(subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain"], cwd=root, text=True, timeout=5).strip())
        return {"runnerCommit": revision, "runnerDirty": dirty, "frontendCommit": "unverified"}
    except (OSError, subprocess.SubprocessError):
        return {"runnerCommit": "unknown", "frontendCommit": "unverified"}


# A marker is document-local. Its disappearance signals reload/new page, while a
# config/lock change in the SAME document is an error, not a silent repair.
_ENSURE = """async ({config, token}) => {
  const api = window.__VISUAL__;
  if (!api) throw new Error('Visual API unavailable');
  const canonical = x => JSON.stringify(x, function(k, v) {
    return v && !Array.isArray(v) && typeof v === 'object'
      ? Object.fromEntries(Object.keys(v).sort().map(k => [k, v[k]])) : v;
  });
  const restored = window.__MOBILEGYM_VISUAL_SESSION__ !== token;
  if (restored) {
    api.unlock(); api.setConfig(config); api.lock();
    window.__MOBILEGYM_VISUAL_SESSION__ = token;
    await document.fonts.ready;
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  }
  const snapshot = api.snapshot();
  if (!snapshot.locked || canonical(snapshot.config) !== canonical(config))
    throw new Error('Visual configuration/lock drift within an episode');
  return {restored, snapshot: restored ? snapshot : null};
}"""

_EXPOSURE = """() => {
  const visible = el => {
    if (!el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0
      && r.top < innerHeight && r.left < innerWidth;
  };
  const collect = attr => [...new Set([...document.querySelectorAll(`[${attr}]`)]
    .filter(visible).map(el => el.getAttribute(attr)))].sort();
  return {apps: collect('data-visual-app'), groups: collect('data-visual-group')};
}"""


class VisualSession:
    """Environment-owned state shared by all runner variants and pool prewarming."""

    def __init__(self, env):
        self.env = env
        self.pending: dict | None = None
        self.active = False
        self.managed_document = False
        self.token = ""
        self.metadata: dict | None = None
        self.observations = 0
        self.apps: Counter = Counter()
        self.groups: Counter = Counter()

    async def reset(self) -> None:
        """End active enforcement, also on the pool's fresh-page reset shortcut."""
        self.active = False
        if self.managed_document:
            # A dead page must not prevent the normal reset recovery from running.
            # On the fresh path, cleanup failure invalidates the shortcut instead.
            try:
                await asyncio.wait_for(self.env.page.evaluate("""() => {
                  const api = window.__VISUAL__;
                  if (!api) throw new Error('Visual API unavailable during cleanup');
                  api.unlock(); api.setProfile('reference');
                  delete window.__MOBILEGYM_VISUAL_SESSION__;
                  const url = new URL(location.href);
                  for (const key of [...url.searchParams.keys()])
                    if (key === 'visual' || key.startsWith('visual_')) url.searchParams.delete(key);
                  history.replaceState(null, '', url);
                }"""), timeout=10)
                self.managed_document = False
            except Exception:
                self.env._fresh = False

    async def begin(self, policy: dict | None, context: dict) -> None:
        await self.reset()
        self.pending = None
        self.metadata = None
        self.observations = 0
        self.apps.clear()
        self.groups.clear()
        if policy is not None:
            validate_visual_target(policy, self.env.url)
            self.pending = resolve_visual_policy(policy, context)
            self.metadata = {**copy.deepcopy(self.pending), "source": visual_source_revision(), "status": "pending", "restores": 0}

    async def activate(self) -> bool:
        if self.pending is None:
            return False
        self.token = uuid.uuid4().hex  # Document ownership only; never a render seed.
        self.active = True
        self.managed_document = True  # Cleanup also if validation/paint fails midway.
        await self.ensure()
        self.metadata["status"] = "active"
        return True

    async def ensure(self) -> None:
        if not self.active:
            return
        try:
            await self.env.page.wait_for_function("() => !!window.__VISUAL__", timeout=30000)
            result = await asyncio.wait_for(self.env.page.evaluate(
                _ENSURE, {"config": self.pending["requestedConfig"], "token": self.token}), timeout=30)
            if result["snapshot"] is not None:
                previous = self.metadata.get("snapshot")
                if previous is not None and previous != result["snapshot"]:
                    raise RuntimeError("Visual resolver/version changed after reload")
                self.metadata["snapshot"] = result["snapshot"]
                self.metadata["restores"] += int(previous is not None)
        except Exception as exc:
            self.metadata["status"] = "error"
            self.metadata["error"] = str(exc)
            raise RuntimeError(f"Visual session failed: {exc}") from exc

    async def observe(self) -> None:
        if self.active:
            try:
                exposure = await asyncio.wait_for(self.env.page.evaluate(_EXPOSURE), timeout=10)
            except Exception as exc:
                self.metadata["status"] = "error"
                self.metadata["error"] = str(exc)
                raise RuntimeError(f"Visual exposure check failed: {exc}") from exc
            self.observations += 1
            self.apps.update(exposure["apps"])
            self.groups.update(exposure["groups"])

    def record(self) -> dict | None:
        if self.metadata is None:
            return None
        # Invalid configs still need a serializable error result. Only inspect axes
        # after the frontend has validated them and returned a resolved snapshot.
        snapshot = self.metadata.get("snapshot")
        theme_apps, layout_groups = [], []
        if snapshot is not None:
            config = snapshot["config"]
            theme_apps = config.get("apps", ["wechat"]) if config["theme"]["strength"] > 0 else []
            if config["version"] == 1:
                layout_groups = ["wechat.primary-nav"] if config["layout"]["strength"] != "native" else []
            else:
                layout_groups = [a["layout"]["group"] for a in snapshot.get("apps", {}).values() if a.get("layoutEnabled")]
        return {
            **copy.deepcopy(self.metadata),
            "exposure": {
                "observations": self.observations,
                "visibleAppObservations": dict(self.apps),
                "visibleLayoutGroupObservations": dict(self.groups),
                "themeCandidateObservations": {a: self.apps[a] for a in theme_apps if self.apps[a]},
                "layoutCandidateObservations": {g: self.groups[g] for g in layout_groups if self.groups[g]},
                "measurement": "DOM visibility at screenshot capture; candidates, not pixel-change or occlusion proof",
            },
        }


async def prepare_visual_episode(env, task, policy=None, context=None) -> None:
    session = getattr(env, "_visual_session", None)
    if policy is None and session is None:
        return  # Default-off: no browser RPCs or extra screenshots.
    if policy is not None and not getattr(env, "supports_state_injection", False):
        raise ValueError("Visual policies require a MobileGym simulator environment")
    if session is None:
        session = env._visual_session = VisualSession(env)
    await session.begin(policy, context or {"mode": "eval", "task_id": task.id, "task_seed": getattr(task, "_seed", None)})


def visual_record(env) -> dict | None:
    session = getattr(env, "_visual_session", None)
    return session.record() if session else None
