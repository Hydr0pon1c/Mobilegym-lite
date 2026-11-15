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

"""Run the vLLM CLI with a scoped startup fix for the pinned PPU image.

vLLM 0.19.0 gives its spawned DP coordinator only 30 seconds to import the
runtime and report socket addresses. Use 120 seconds, as newer upstream
versions do. This patch is process-local; it does not modify site-packages.
Remove it when the accelerator image moves off 0.19.0.
"""

from importlib.metadata import version
import multiprocessing.connection


def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
    try:
        ready = multiprocessing.connection.wait(
            [zmq_addr_pipe, self.proc.sentinel], timeout=120
        )
        if not ready:
            raise RuntimeError(
                "DP Coordinator did not report ZMQ addresses within 120 seconds "
                f"during startup (pid={self.proc.pid})."
            )
        if zmq_addr_pipe not in ready:
            raise RuntimeError("DP Coordinator process exited during startup.")
        try:
            return zmq_addr_pipe.recv()
        except EOFError:
            raise RuntimeError("DP Coordinator process failed during startup.") from None
    finally:
        zmq_addr_pipe.close()


def main() -> None:
    from vllm.entrypoints.cli.main import main as vllm_main

    if version("vllm").split("+", 1)[0] == "0.19.0":
        from vllm.v1.engine.coordinator import DPCoordinator

        DPCoordinator._wait_for_zmq_addrs = _wait_for_zmq_addrs
        print("[mobilegym] vLLM 0.19.0 DP coordinator startup timeout: 120s", flush=True)

    vllm_main()


if __name__ == "__main__":
    main()
