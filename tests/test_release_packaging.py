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

"""Source ZIP boundaries, privacy checks, and operation without Git."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from scripts import check_release as release


@pytest.fixture
def source(tmp_path, monkeypatch):
    original = release.ROOT
    root = tmp_path / 'source'
    root.mkdir()
    for relative in (
        '.gitignore', 'configs/training/full.json',
        'configs/training/feature0.json', 'configs/training/feature0-selective.json',
        'bench_env/splits/env_effect_v3_train.txt', 'bench_env/splits/env_effect_v3_test.txt',
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)
    monkeypatch.setattr(release, 'ROOT', root)
    return root


def put(root, name, data=b'fixture\n'):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def test_git_and_extracted_inventory_agree_after_cleanup(source):
    # A source package named models must survive; generated model weights must not.
    put(source, 'mobilegym-rl/verl/verl/models/policy.py')
    deleted = put(source, 'retired-report.txt')
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    deleted.unlink()
    for name in (
        '.venv-release/lib/dependency.py', 'artifacts/run/results.jsonl',
        'mobilegym-rl/models/weights.safetensors', '__pycache__/cache.pyc',
        '.nginx/nginx.pid', 'public/sdcard/manifest.json', 'public/app_nav_graph.json',
    ):
        put(source, name)
    git_files = release.source_files()
    shutil.rmtree(source / '.git')
    assert release.source_files() == git_files
    assert 'mobilegym-rl/verl/verl/models/policy.py' in git_files
    assert 'retired-report.txt' not in git_files
    assert release.check(git_files)[0] == []


@pytest.mark.parametrize('data', [
    b'\x00' + b'/' + b'home' + b'/researcher/private/experiment',
    b'/' + b'data0' + b'/researcher/models',
    b'C:' + b'\\Users' + b'\\researcher\\experiment',
])
def test_personal_paths_are_rejected_including_binary_metadata(source, data):
    put(source, 'asset.bin', data)
    errors, _ = release.check(release.source_files())
    assert 'Machine-specific research path: asset.bin' in errors


@pytest.mark.parametrize('token', [
    b'sk-proj-' + b'x' * 40,
    b'ghp_' + b'x' * 36,
    b'AIza' + b'x' * 35,
])
def test_credentials_are_rejected(source, token):
    put(source, 'config.txt', token)
    errors, _ = release.check(release.source_files())
    assert 'Potential credential in source: config.txt' in errors


def test_notebook_execution_output_is_rejected(source):
    notebook = {'cells': [{'cell_type': 'code', 'execution_count': 1,
                           'source': ['print(1)'], 'outputs': [{'text': '1'}]}]}
    put(source, 'example.ipynb', json.dumps(notebook).encode())
    errors, _ = release.check(release.source_files())
    assert 'Notebook execution output in source: example.ipynb' in errors


def test_symlink_directory_is_rejected_without_git(source, tmp_path):
    (source / 'linked').symlink_to(tmp_path, target_is_directory=True)
    errors, _ = release.check(release.source_files())
    assert 'Artifact symlink in source: linked' in errors


def test_zip_uses_only_verified_inventory_and_neutral_metadata(source, tmp_path):
    script = put(source, 'run.sh', b'#!/bin/sh\nexit 0\n')
    script.chmod(0o755)
    errors, hashes = release.check(release.source_files())
    assert errors == []
    put(source, release.MANIFEST, json.dumps({'algorithm': 'sha256', 'files': hashes}).encode())
    put(source, '.git/config', b'private history')
    put(source, '.env', b'private setting')
    put(source, 'artifacts/run/results.jsonl')
    output = tmp_path / 'release.zip'
    release.write_zip(output, hashes)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            f'Mobilegym-lite/{name}' for name in {*hashes, release.MANIFEST}
        }
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(f'Mobilegym-lite/{name}')).hexdigest() == digest
        for info in archive.infolist():
            assert not info.extra and not info.comment
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert archive.getinfo('Mobilegym-lite/run.sh').external_attr >> 16 & 0o777 == 0o755
    with pytest.raises(ValueError, match='outside the source'):
        release.write_zip(source / 'release.zip', hashes)
