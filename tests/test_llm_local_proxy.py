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

import pytest

from bench_env.llm.openai_chat import LLMClient, _is_local_endpoint


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
        "http://10.0.0.2:8000/v1",
        "http://192.168.1.2:8000/v1",
    ],
)
def test_local_endpoint_detection(url):
    assert _is_local_endpoint(url)


def test_public_endpoint_detection():
    assert not _is_local_endpoint("https://api.openai.com/v1")


def test_local_client_ignores_environment_proxy(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks5h://proxy.invalid:1080")
    client = LLMClient(base_url="http://10.0.0.2:8000/v1", model="test")
    client._client.close()
