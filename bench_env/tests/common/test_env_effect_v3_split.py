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

from __future__ import annotations

import csv
import json
from pathlib import Path

from bench_env.splits import resolve_split


ROOT = Path(__file__).resolve().parents[3]
SPLIT_ROOT = ROOT / "bench_env" / "splits"
PREFIX = "env_effect_v3"


def _ids(name: str) -> set[str]:
    return resolve_split(f"{PREFIX}_{name}")


def test_env_effect_v3_relationships() -> None:
    official = resolve_split("train") | resolve_split("test")
    core = _ids("core")
    shared = _ids("shared")
    train_only = _ids("train_only")
    ood_test = _ids("ood_test")
    train = _ids("train")
    test = _ids("test")
    stress = _ids("stress")
    excluded = _ids("excluded")

    assert len(official) == 416
    assert len(core) == 215
    assert len(shared) == 132
    assert len(train_only) == 60
    assert len(ood_test) == 23
    assert len(train) == 192
    assert len(test) == 155
    assert len(stress) == 106
    assert len(excluded) == 95

    assert train & test == shared
    assert train - test == train_only
    assert test - train == ood_test
    assert core == train | test
    assert not (core & excluded)
    assert not (core & stress)
    assert not (excluded & stress)
    assert core | excluded | stress == official


def test_env_effect_v3_manifest_matches_active_profile() -> None:
    manifest = json.loads(
        (SPLIT_ROOT / f"{PREFIX}_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 2
    assert manifest["relationships"] == {
        "test_only_ood": 23,
        "test_template_ood_fraction": 23 / 155,
        "train_only": 60,
        "train_test_overlap": 132,
    }

    sampling = manifest["recommended_sampling"]
    assert sampling["train_sample_n"] == 1
    assert sampling["train_task_seed"] == 41
    assert sampling["test_sample_n"] == 1
    assert sampling["test_task_seeds"] == [42, 43, 44]
    assert sampling["train_batch_size"] == 12
    assert sampling["optimizer_steps_per_epoch"] == 16

    for name, spec in manifest["sets"].items():
        assert spec["file"] == f"{PREFIX}_{name}.txt"
        assert spec["templates"] == len(_ids(name))


def test_env_effect_v3_candidate_csv_matches_assignments() -> None:
    with (SPLIT_ROOT / f"{PREFIX}_candidates.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["task_id", "assignment"]
        rows = list(reader)

    assert len(rows) == 416
    assert len({row["task_id"] for row in rows}) == 416
    for assignment in ("shared", "train_only", "ood_test", "stress", "excluded"):
        assert {
            row["task_id"] for row in rows if row["assignment"] == assignment
        } == _ids(assignment)
