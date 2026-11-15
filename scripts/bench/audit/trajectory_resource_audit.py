#!/usr/bin/env python3
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

"""Audit HTTP/resource failures recorded in MobileGym eval browser logs.

The report keeps absence, invalid runtime paths, loading failures, remote HTTP
errors, and cancelled requests separate.  It is intentionally diagnostic: it
does not download files or mutate override maps.  Confirmed remote failures can
then be passed to ``scripts/dev/localize-external-assets.py`` for repair.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Iterable
from urllib.parse import unquote, urlsplit


HTTP_ERROR_RE = re.compile(
    r"(?:\[(?P<task>[^\]]+)\]\s+)?\[page#(?P<page>\d+)\]\s+"
    r"http_error:\s*(?P<status>\d{3})\s+[—-]\s+(?P<url>\S.*)$"
)
REQUEST_FAILED_RE = re.compile(
    r"(?:\[(?P<task>[^\]]+)\]\s+)?\[page#(?P<page>\d+)\]\s+"
    r"requestfailed:\s*(?P<failure>.*?)\s+[—-]\s+(?P<url>\S.*)$"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EXTENSION_ONLY_RE = re.compile(r"\.(?:avif|bmp|gif|jpe?g|png|svg|webm?|mp4)$", re.I)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

ACTIONABLE_CLASSIFICATIONS = {
    "dynamic_invalid_path",
    "external_missing_resource",
    "local_404_unresolved",
    "local_missing_resource",
    "local_missing_suspected",
    "local_route_or_loading",
    "map_offline_cache_miss",
}
LOCAL_ACTIONABLE_CLASSIFICATIONS = {
    value for value in ACTIONABLE_CLASSIFICATIONS if not value.startswith("external_")
}


@dataclass(frozen=True)
class Observation:
    source: str
    line: int
    task: str | None
    page: int
    event: str
    url: str
    status: int | None = None
    failure: str | None = None


@dataclass(frozen=True)
class Classification:
    name: str
    scope: str
    actionable: bool
    route_kind: str | None
    local_candidates: tuple[str, ...]
    local_exists: bool | None
    repair_hint: str


def discover_log_files(paths: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        path = raw.resolve()
        if path.is_file():
            found.add(path)
            continue
        if not path.is_dir():
            continue
        if path.name == "browser_logs":
            found.update(item.resolve() for item in path.rglob("*.log"))
            continue
        direct_browser_logs = path / "browser_logs"
        if direct_browser_logs.is_dir():
            found.update(item.resolve() for item in direct_browser_logs.rglob("*.log"))
            continue
        nested = list(path.rglob("browser_logs/*.log"))
        if nested:
            found.update(item.resolve() for item in nested)
        else:
            found.update(item.resolve() for item in path.glob("*.log"))
    return sorted(found)


def parse_log_file(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with path.open(errors="ignore") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = ANSI_RE.sub("", raw_line.rstrip())
            http_match = HTTP_ERROR_RE.search(line)
            if http_match:
                observations.append(Observation(
                    source=str(path),
                    line=line_number,
                    task=http_match.group("task"),
                    page=int(http_match.group("page")),
                    event="http_error",
                    status=int(http_match.group("status")),
                    failure=None,
                    url=http_match.group("url").strip(),
                ))
                continue
            failed_match = REQUEST_FAILED_RE.search(line)
            if failed_match:
                observations.append(Observation(
                    source=str(path),
                    line=line_number,
                    task=failed_match.group("task"),
                    page=int(failed_match.group("page")),
                    event="requestfailed",
                    status=None,
                    failure=failed_match.group("failure").strip(),
                    url=failed_match.group("url").strip(),
                ))
    return observations


def _is_dynamic_invalid_path(url: str) -> bool:
    path = unquote(urlsplit(url).path).rstrip()
    basename = PurePosixPath(path).name
    return not basename or bool(EXTENSION_ONLY_RE.fullmatch(basename))


def _local_candidates(url: str, repo_root: Path) -> tuple[str | None, list[Path]]:
    path = unquote(urlsplit(url).path)
    parts = PurePosixPath(path).parts
    if len(parts) >= 4 and parts[1] == "@app-assets":
        app_name = parts[2]
        relative = Path(*parts[3:])
        candidates: list[Path] = []
        for base in ("apps", "system"):
            app_root = repo_root / base / app_name
            target = (
                app_root / "wmr" / Path(*parts[4:])
                if parts[3] == "wmr" and len(parts) >= 5
                else app_root / "assets" / relative
            )
            candidates.append(target)
        return "app_asset", candidates
    if len(parts) >= 3 and parts[1] == "cdn":
        return "cdn", [repo_root / "mobilegym-data" / Path(*parts[2:])]
    if len(parts) >= 2 and parts[1] == "sdcard":
        return "sdcard", [repo_root / "public" / Path(*parts[1:])]
    if len(parts) >= 3 and parts[1] == "map-cache":
        return "map_cache", [repo_root / "apps" / "Map" / "sw" / "files" / Path(*parts[2:])]
    if len(parts) >= 3 and parts[1] == "map-vector-cache":
        return "map_vector_cache", [repo_root / "apps" / "Map" / "sw" / "vector" / Path(*parts[2:])]
    if len(parts) >= 2 and parts[1] not in {"api", "@vite", "src"}:
        return "public", [repo_root / "public" / Path(*parts[1:])]
    return None, []


def classify(observation: Observation, repo_root: Path) -> Classification:
    parsed = urlsplit(observation.url)
    is_http = parsed.scheme in {"http", "https"}
    is_local = is_http and (parsed.hostname or "").lower() in LOCAL_HOSTS
    route_kind, candidates = _local_candidates(observation.url, repo_root) if is_local else (None, [])
    candidate_strings = tuple(str(path) for path in candidates)
    local_exists = any(path.is_file() for path in candidates) if candidates else None

    def result(name: str, hint: str, *, actionable: bool | None = None) -> Classification:
        return Classification(
            name=name,
            scope="local" if is_local else "external",
            actionable=name in ACTIONABLE_CLASSIFICATIONS if actionable is None else actionable,
            route_kind=route_kind,
            local_candidates=candidate_strings,
            local_exists=local_exists,
            repair_hint=hint,
        )

    if not is_http:
        return result(
            "unsupported_scheme",
            "Usually belongs to an external page/app deep link; inspect only if the native task depends on it.",
            actionable=False,
        )

    if observation.event == "http_error":
        if observation.status in {404, 410}:
            if not is_local:
                return result(
                    "external_missing_resource",
                    "Retry the exact URL, then localize it or add an exact negative override only when 404/410 is confirmed.",
                )
            if _is_dynamic_invalid_path(observation.url):
                return result(
                    "dynamic_invalid_path",
                    "Fix the expression/default that generated a filename-less path; do not add a placeholder file for it.",
                )
            if candidates and not local_exists:
                return result(
                    "local_missing_resource",
                    "The URL maps to known repository/data locations and no file exists; restore the asset or remove the request.",
                )
            if candidates and local_exists:
                return result(
                    "local_route_or_loading",
                    "The backing file exists; inspect URL decoding, Vite/CDN routing, permissions, and initialization order.",
                )
            return result(
                "local_404_unresolved",
                "The local URL is not a known static route; inspect its server/plugin handler.",
            )
        if (
            observation.status == 504
            and (parsed.hostname or "").lower() == "maps.googleapis.com"
            and (
                parsed.path == "/maps/vt"
                or parsed.path.startswith("/maps/vt/pb=")
            )
        ):
            return result(
                "map_offline_cache_miss",
                "Run apps/Map/scripts/backfill_vector_tiles.mjs on the eval run. "
                "It will refuse to merge tiles when the bundled SDK/config epoch is incompatible.",
            )
        return result(
            "local_http_error" if is_local else "external_http_error",
            "Inspect the HTTP status and owning endpoint; this is not evidence that an asset is missing.",
            actionable=is_local,
        )

    failure = (observation.failure or "").upper()
    if "ERR_ABORTED" in failure:
        if is_local and candidates and not local_exists:
            return result(
                "local_missing_suspected",
                "The request was aborted and its known backing file is absent; reproduce once to obtain a definitive HTTP status.",
            )
        return result(
            "local_request_cancelled" if is_local else "external_request_cancelled",
            "Usually caused by navigation or request replacement; reproduce only if the UI visibly failed.",
            actionable=False,
        )
    if is_local:
        if candidates and not local_exists:
            return result(
                "local_missing_suspected",
                "The request failed and its known backing file is absent; restore it or remove the request after reproduction.",
            )
        if candidates and local_exists:
            return result(
                "local_loading_failure",
                "The file exists, so inspect routing, decoding, media parsing, or cancellation rather than adding an asset.",
                actionable=True,
            )
        return result(
            "local_request_failure",
            "Inspect the owning local endpoint and reproduce the request.",
            actionable=True,
        )
    return result(
        "external_transient_failure",
        "Retry separately; timeouts, resets, blocking, and CORS/ORB failures are not proof of a missing resource.",
        actionable=False,
    )


def build_report(
    inputs: list[Path],
    log_files: list[Path],
    observations: list[Observation],
    repo_root: Path,
) -> dict[str, object]:
    grouped: dict[tuple[object, ...], list[tuple[Observation, Classification]]] = defaultdict(list)
    classifications = Counter()
    statuses = Counter()
    tasks: set[str] = set()

    for observation in observations:
        classification = classify(observation, repo_root)
        classifications[classification.name] += 1
        if observation.status is not None:
            statuses[str(observation.status)] += 1
        if observation.task:
            tasks.add(observation.task)
        key = (
            classification.name,
            observation.status,
            observation.failure,
            observation.url,
        )
        grouped[key].append((observation, classification))

    issues: list[dict[str, object]] = []
    for entries in grouped.values():
        first_observation, first_classification = entries[0]
        locations = [
            {
                "source": observation.source,
                "line": observation.line,
                "task": observation.task,
                "page": observation.page,
            }
            for observation, _ in entries
        ]
        issues.append({
            "classification": first_classification.name,
            "scope": first_classification.scope,
            "actionable": first_classification.actionable,
            "status": first_observation.status,
            "failure": first_observation.failure,
            "url": first_observation.url,
            "count": len(entries),
            "tasks": sorted({item[0].task for item in entries if item[0].task}),
            "route_kind": first_classification.route_kind,
            "local_candidates": list(first_classification.local_candidates),
            "local_exists": first_classification.local_exists,
            "repair_hint": first_classification.repair_hint,
            "locations": locations,
        })

    issues.sort(key=lambda issue: (
        not bool(issue["actionable"]),
        str(issue["classification"]),
        str(issue["url"]),
    ))
    return {
        "schema_version": 1,
        "repo_root": str(repo_root),
        "inputs": [str(path.resolve()) for path in inputs],
        "log_files": len(log_files),
        "tasks_observed": len(tasks),
        "occurrences": len(observations),
        "unique_issues": len(issues),
        "actionable_occurrences": sum(
            count for name, count in classifications.items() if name in ACTIONABLE_CLASSIFICATIONS
        ) + sum(
            count for name, count in classifications.items()
            if name in {"local_http_error", "local_loading_failure", "local_request_failure"}
        ),
        "summary": {
            "by_classification": dict(sorted(classifications.items())),
            "by_http_status": dict(sorted(statuses.items())),
        },
        "issues": issues,
    }


def render_text(report: dict[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, dict)
    by_classification = summary["by_classification"]
    assert isinstance(by_classification, dict)
    lines = [
        "Trajectory resource audit",
        f"  logs: {report['log_files']}",
        f"  tasks: {report['tasks_observed']}",
        f"  occurrences: {report['occurrences']}",
        f"  unique issues: {report['unique_issues']}",
        f"  actionable occurrences: {report['actionable_occurrences']}",
        "  classifications: " + ", ".join(f"{key}={value}" for key, value in by_classification.items()),
        "",
    ]
    issues = report["issues"]
    assert isinstance(issues, list)
    for issue in issues:
        assert isinstance(issue, dict)
        marker = "ACTION" if issue["actionable"] else "INFO"
        status = issue["status"] or issue["failure"] or "failure"
        lines.append(
            f"[{marker}] {issue['classification']} x{issue['count']} ({status}) {issue['url']}"
        )
        if issue["tasks"]:
            lines.append("  tasks: " + ", ".join(issue["tasks"]))
        if issue["local_candidates"]:
            exists = "exists" if issue["local_exists"] else "missing"
            lines.append(f"  local target ({exists}): " + ", ".join(issue["local_candidates"]))
        lines.append(f"  next: {issue['repair_hint']}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="run roots, browser_logs directories, or individual .log files")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write the report to a file instead of stdout")
    parser.add_argument("--http-only", action="store_true", help="exclude requestfailed events")
    parser.add_argument("--fail-on-local", action="store_true", help="exit 1 when a local actionable issue is present")
    args = parser.parse_args(argv)

    log_files = discover_log_files(args.paths)
    if not log_files:
        parser.error("no browser log files found under the supplied paths")

    observations = [item for path in log_files for item in parse_log_file(path)]
    if args.http_only:
        observations = [item for item in observations if item.event == "http_error"]
    report = build_report(args.paths, log_files, observations, args.repo_root.resolve())
    rendered = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else render_text(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)

    if args.fail_on_local:
        issues = report["issues"]
        assert isinstance(issues, list)
        if any(
            isinstance(issue, dict)
            and issue.get("classification") in LOCAL_ACTIONABLE_CLASSIFICATIONS | {
                "local_http_error", "local_loading_failure", "local_request_failure"
            }
            for issue in issues
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
