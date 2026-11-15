# Copyright 2026 OPPO
#
# Modified by OPPO for this release.
# OPPO copyright applies only to the modifications.
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

import re
from typing import Any

from bench_env.task.common_tasks import AnswerTask, build_answer_checks
from bench_env.task.judge import JudgeInput
from bench_env.task.map.app import MAP_SEARCH_CHANGES


class NorthResearchInstituteAnswer(AnswerTask):
    """Verdict: answer the closest institute shown northeast of the device."""

    templates = [
        "我所在位置东北方向最近的研究所是什么",
        "请查看地图，告诉我当前位置东北边最近的研究所叫什么",
    ]
    apps = ["map"]
    scope = "S2"
    objective = "query"
    composition = "deep_dive"
    difficulty = "L4"
    capabilities = ["search", "extract", "reasoning"]
    expected_changes = MAP_SEARCH_CHANGES
    # Consumed by apps/Map/scripts/fetch_geo_snapshot.mjs.
    map_search_queries = ["研究所"]

    answer_fields = [
        {
            "type": "text",
            "label": "研究所名称",
            "hint": "填写研究所名称",
        }
    ]
    answer = re.compile(r"中国科学院自动化研究所|中科院自动化所|自动化所")

    def check_goals(self, input: JudgeInput) -> list[dict[str, Any]]:
        checks = build_answer_checks(self.get_answer(input), input.answer)
        answer = str(input.answer or "")
        negated = bool(re.search(r"(?:不是|并不是|并非|非).{0,8}(?:自动化所|自动化研究所)", answer))
        if negated:
            checks[0]["passed"] = False
            checks[0]["reason"] = "answer contains a negated allowed institute name"
        return checks
