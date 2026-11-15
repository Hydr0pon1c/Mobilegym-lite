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

"""Validate the public source boundary and optionally refresh its SHA-256 manifest."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "release-manifest.json"
IGNORED_DIRS = {
    ".git", "node_modules", "dist", "dist-ssr", "__pycache__", ".pytest_cache",
    ".idea", ".vscode", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints", "wandb",
}
RUNTIME_DIRS = {
    "mobilegym-data", "artifacts", ".rllm_local", "logs", "runs", "models",
    "checkpoints", "outputs", "paper", "reports", "swanlog", "eval_results",
}
PRIVATE_PATH = re.compile(
    rb"(?<![\w.])/(?:home|Users|data[0-9]+|scratch)/[^\s/\"'<>`]+/"
    rb"|[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/\"'<>`]+[\\/]"
)
OFFLINE_MAP_KEY = b"AIzaSyOfflineMapCacheOnly00000000000000"
SECRETS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"(?<![\w-])sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}"),
    re.compile(rb"(?<!\w)hf_[A-Za-z0-9]{30,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
)
SECRET_MARKERS = (b"-----BEGIN", b"gh", b"github_pat_", b"sk-", b"hf_", b"xox", b"AKIA", b"AIza")


def ignored_directory(relative):
    parts = Path(relative).parts
    if not parts:
        return False
    return (
        any(name in IGNORED_DIRS or name.startswith((".venv", "venv"))
            or name.endswith(".egg-info") for name in parts)
        or parts[0] in RUNTIME_DIRS
        or (len(parts) > 1 and parts[0] == "mobilegym-rl" and parts[1] in RUNTIME_DIRS)
    )


def generated_file(name):
    path = Path(name)
    return (
        name == MANIFEST
        or path.name in {".DS_Store", ".api_gateway.pid", ".serve_api.pid", ".coverage"}
        or path.suffix in {".pyc", ".pyo", ".log", ".tsbuildinfo"}
        or name == "index.css"
        or (name.startswith(".nginx/") and name != ".nginx/nginx.source.conf")
        or name == "public/sdcard/manifest.json"
        or any(fnmatch.fnmatchcase(name, pattern) for pattern in (
            "public/*_nav_graph*.json", "public/*_data_graph.json",
            "public/*_action_tasks*.json", "apps/*/data/.bench_*_cache.json",
        ))
    )


def source_files():
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT, check=True, capture_output=True,
        )
        names = set(result.stdout.decode().strip("\0").split("\0")) - {"", MANIFEST}
        # git ls-files still lists tracked files deleted from the working tree.
        return sorted(name for name in names
                      if (ROOT / name).is_file() or (ROOT / name).is_symlink())
    paths = []
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        relative_directory = Path(directory).relative_to(ROOT)
        # Reject symlinks during check() instead of silently omitting them.
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink() and not ignored_directory(relative_directory / name):
                paths.append(path.relative_to(ROOT).as_posix())
        dirs[:] = [name for name in dirs if not ignored_directory(relative_directory / name)
                   and not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(ROOT).as_posix()
            if generated_file(relative):
                continue
            paths.append(relative)
    return sorted(paths)


def check(files):
    errors = []
    hashes = {}
    for name in files:
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"Artifact symlink in source: {name}")
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            errors.append(f"Credential file in source: {name}")
        if path.suffix.lower() in {".key", ".pem", ".safetensors", ".ckpt", ".pt", ".pth"}:
            errors.append(f"Key or model artifact in source: {name}")
        if ignored_directory(Path(name).parent):
            errors.append(f"Runtime directory in source: {name}")
        if generated_file(name):
            errors.append(f"Generated runtime file in source: {name}")
        data = path.read_bytes()
        if (any(marker in data for marker in (b"/home/", b"/Users/", b"/data", b"/scratch/", b"Users"))
                and PRIVATE_PATH.search(data)):
            errors.append(f"Machine-specific research path: {name}")
        # This deliberately invalid sentinel identifies the bundled offline map cache.
        credential_data = data.replace(OFFLINE_MAP_KEY, b"")
        if any(marker in credential_data and pattern.search(credential_data)
               for marker, pattern in zip(SECRET_MARKERS, SECRETS)):
            errors.append(f"Potential credential in source: {name}")
        if path.suffix == ".ipynb":
            notebook = json.loads(data)
            if any(cell.get("outputs") or cell.get("execution_count") is not None
                   for cell in notebook.get("cells", [])):
                errors.append(f"Notebook execution output in source: {name}")
        hashes[name] = hashlib.sha256(data).hexdigest()
    modes = {"full": None, "feature0": "feature0", "feature0-selective": "feature0_selective_v1"}
    for name, mode in modes.items():
        spec = json.loads((ROOT / f"configs/training/{name}.json").read_text())
        if spec["overrides"]["train_completeness"] != mode or spec["overrides"]["val_completeness"] is not None:
            errors.append(f"Invalid train/Full-validation condition: {name}")
        if {"MODEL_PATH", "MOBILEGYM_STATIC_DIR", "RUN_NAME"} & spec["environment"].keys():
            errors.append(f"Machine-specific environment defaults: {name}")
    for split, expected in (("train", 192), ("test", 155)):
        lines = (ROOT / f"bench_env/splits/env_effect_v3_{split}.txt").read_text().splitlines()
        ids = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if len(ids) != expected or len(set(ids)) != expected:
            errors.append(f"Incorrect {split} template membership")
    return errors, hashes


def write_zip(output, hashes):
    output = Path(output).expanduser().resolve()
    if output.is_relative_to(ROOT.resolve()) or output.suffix.lower() != ".zip":
        raise ValueError("Choose a .zip output path outside the source directory")
    # Fixed timestamps and permission-only attributes avoid local user metadata.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name in sorted([*hashes, MANIFEST]):
            path = ROOT / name
            data = path.read_bytes()
            if name != MANIFEST and hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError(f"Source changed during packaging: {name}")
            info = zipfile.ZipInfo(f"Mobilegym-lite/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    print(f"Source ZIP created: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Refresh hashes after intentional source changes")
    parser.add_argument("--zip", metavar="PATH", help="Package the verified source inventory outside the checkout")
    args = parser.parse_args()
    errors, hashes = check(source_files())
    manifest_path = ROOT / MANIFEST
    if not args.write_manifest:
        if not manifest_path.exists():
            errors.append("Missing release-manifest.json; use --write-manifest after reviewing source")
        else:
            recorded = json.loads(manifest_path.read_text())["files"]
            for path in sorted(recorded.keys() | hashes.keys()):
                if recorded.get(path) != hashes.get(path):
                    errors.append(f"Inventory differs: {path}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if args.write_manifest:
        manifest_path.write_text(json.dumps({"algorithm": "sha256", "files": hashes}, indent=2) + "\n")
    print(f"Release checks passed: {len(hashes)} source files; train192/test155; Full/feature0/selective")
    if args.zip:
        try:
            write_zip(args.zip, hashes)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
