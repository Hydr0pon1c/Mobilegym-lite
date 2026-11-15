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

"""Backfill map vector-tile cache misses observed in browser logs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


MISS_MARKER = "offline-only cache miss "
ALLOWED_HOSTS = {"maps.googleapis.com"}


def normalize_tile_pb(value: str) -> str:
    value = re.sub(r"(!2sm!3i)\d+", r"\1E", value)
    value = re.sub(r"(!28i)\d+", r"\1E", value)
    return re.sub(r"!5m2!1e3!5f\d+", "!5m1!1e3", value)


def clean_url(raw: str) -> str | None:
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_HOSTS:
        return None
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
             if key not in {"key", "token", "callback"}]
    path = parsed.path
    if path == "/maps/vt":
        pb = next((value for key, value in query if key == "pb"), None)
        if pb is not None:
            path = f"/maps/vt/pb={normalize_tile_pb(pb)}"
            query = [(key, value) for key, value in query if key != "pb"]
    elif path.startswith("/maps/vt/pb="):
        path = normalize_tile_pb(path)
    if not path.startswith("/maps/vt/pb="):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(sorted(query)), ""))


def urls_from_logs(paths: list[Path]) -> set[str]:
    urls: set[str] = set()
    for root in paths:
        files = root.rglob("*.log") if root.is_dir() else [root]
        for path in files:
            try:
                with path.open(errors="ignore") as stream:
                    for line in stream:
                        if MISS_MARKER not in line:
                            continue
                        raw = line.split(MISS_MARKER, 1)[1].split()[0].rstrip("\"',;:)]}>")
                        normalized = clean_url(raw)
                        if normalized:
                            urls.add(normalized)
            except OSError:
                continue
    return urls


def extension_for(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    if media_type == "image/jpeg":
        return ".jpg"
    return ".bin"


def download_entry(
    url: str,
    files_dir: Path,
    *,
    insecure: bool,
    timeout: int,
) -> tuple[dict[str, object] | None, str]:
    with tempfile.TemporaryDirectory(prefix="map-cache-") as tmp:
        body_path = Path(tmp) / "body"
        command = [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--max-time",
            str(timeout),
            "--output",
            str(body_path),
            "--write-out",
            "%{http_code}\n%{content_type}\n",
        ]
        if insecure:
            command.append("--insecure")
        command.append(url)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        lines = completed.stdout.splitlines()
        status = lines[0].strip() if lines else "curl_error"
        content_type = lines[1].strip() if len(lines) >= 2 else "application/octet-stream"
        if completed.returncode != 0 or status != "200" or not body_path.is_file():
            return None, status
        body = body_path.read_bytes()
        if not body:
            return None, "empty_body"
        filename = f"{hashlib.sha1(url.encode()).hexdigest()[:16]}{extension_for(content_type)}"
        destination = files_dir / filename
        if not destination.exists():
            shutil.copyfile(body_path, destination)
        return {
            "url": url,
            "file": filename,
            "contentType": content_type,
            "status": 200,
            "size": len(body),
            "bodyHash": hashlib.sha1(body).hexdigest(),
        }, "downloaded"


def cache_name(entries: list[dict[str, object]]) -> str:
    rows = []
    for entry in entries:
        rows.append("\t".join(str(entry.get(key, "") or "") for key in (
            "url", "file", "status", "size", "bodyHash",
        )))
    digest = hashlib.sha1("\n".join(rows).encode()).hexdigest()[:12]
    return f"map-vector-cache-{digest}"


def atomic_write_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def update_runtime_metadata(repo_root: Path, next_cache_name: str) -> None:
    sw_path = repo_root / "apps/Map/sw/map-sw.js"
    source = sw_path.read_text()
    pattern = re.compile(r"(const VECTOR_CACHE_NAME = 'map-vector-cache-[0-9a-f]{12})([^']*)(';)")
    updated, count = pattern.subn(
        lambda match: f"const VECTOR_CACHE_NAME = '{next_cache_name}{match.group(2)}';",
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not update VECTOR_CACHE_NAME in map-sw.js")
    sw_path.write_text(updated)

    bootstrap_path = repo_root / "apps/Map/sw/vector/bootstrap-manifest.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    bootstrap["sourceCacheName"] = next_cache_name
    atomic_write_json(bootstrap_path, bootstrap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("browser_logs", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("mobilegym-data/map/vector/manifest.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = (repo_root / args.manifest).resolve()
    files_dir = manifest_path.parent / "files"
    manifest = json.loads(manifest_path.read_text())
    entries = list(manifest.get("entries") or [])
    existing = {clean_url(str(entry.get("url") or "")) for entry in entries}
    discovered = urls_from_logs([path.resolve() for path in args.browser_logs])
    pending = sorted(url for url in discovered if url not in existing)

    downloaded: list[dict[str, object]] = []
    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download_entry,
                url,
                files_dir,
                insecure=args.insecure,
                timeout=args.timeout,
            ): url
            for url in pending
        }
        for future in as_completed(futures):
            entry, status = future.result()
            statuses[status] = statuses.get(status, 0) + 1
            if entry:
                downloaded.append(entry)

    if downloaded:
        by_url = {str(entry["url"]): entry for entry in entries}
        by_url.update({str(entry["url"]): entry for entry in downloaded})
        sorted_entries = sorted(by_url.values(), key=lambda entry: str(entry["url"]))
        next_cache_name = cache_name(sorted_entries)
        manifest.update({
            "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cacheName": next_cache_name,
            "entries": sorted_entries,
        })
        atomic_write_json(manifest_path, manifest)
        update_runtime_metadata(repo_root, next_cache_name)

    print(json.dumps({
        "discovered": len(discovered),
        "already_cached": len(discovered) - len(pending),
        "downloaded": len(downloaded),
        "statuses": statuses,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
