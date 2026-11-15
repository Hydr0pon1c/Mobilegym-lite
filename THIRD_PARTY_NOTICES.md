# Third-party notices

This release is derived from MobileGym and vendors its training dependencies.
Original copyright and license headers remain in the source files.

New code contributed for this research release is Copyright 2026 OPPO and is
licensed under Apache-2.0. Files modified from the recorded MobileGym v0.1.0
upstream revision carry an OPPO notice that covers only the modifications;
their original copyright and license notices are retained. Unmodified upstream
files do not receive an OPPO notice. The exact file scope and upstream revision
are recorded in [copyright-scope.json](copyright-scope.json).

| Component | Upstream | Included license |
| --- | --- | --- |
| MobileGym simulator and benchmark | https://github.com/Purewhiter/mobilegym | [Apache-2.0](LICENSE) |
| MobileGym data and snapshots | Same upstream; third-party provenance varies | [CC BY-NC 4.0](LICENSE-DATA), [DISCLAIMER](DISCLAIMER.md) |
| rLLM training stack | https://github.com/rllm-org/rllm | [Apache-2.0](mobilegym-rl/LICENSE) |
| VERL | https://github.com/volcengine/verl | [Apache-2.0](mobilegym-rl/verl/LICENSE), [notice](mobilegym-rl/verl/Notice.txt) |
| rLLM model gateway | Included in the rLLM source distribution | Apache-2.0, as declared in its [package metadata](mobilegym-rl/rllm-model-gateway/pyproject.toml) |

The frontend package lock records dependency versions. External Python, browser,
accelerator, model, and companion-data distributions retain their respective
licenses. This source release does not redistribute model weights or the
separately downloaded companion dataset.

Release modifications include portable training/evaluation paths, public example
configurations, installation guidance, scoped documentation, and release checks.
The vendored rLLM/VERL source also includes the retained MobileGym training,
padding, seed, export, clip-schedule, and optional episode-weighting integration.
