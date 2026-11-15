# Repository instructions

Respond in the user's language. This release contains the MobileGym simulator,
state-judged benchmark, and vendored rLLM/VERL training stack.

- Start with `README.md`, `docs/feature-pruning.md`, and `mobilegym-rl/README.md`.
- Keep Full, feature0, and `feature0_selective_v1` explicit. The paper's Lite name
  means selective, not the internal `lite_v1` resource-plus-feature mode.
- Keep feature plans in `bench_env/feature_completeness.py` and shared frontend
  gates in `os/featureAvailability.tsx`. Apps consume module IDs, not task IDs.
- Preserve required data, both conditional branches, alternate entrances, native
  judges, and shared helpers. Unknown templates and standard evaluation stay Full.
- Preserve default GSPO clip 0.2/0.28 and response_mean episode weighting.
- Use `MOBILEGYM_CONFIG_ONLY=1` for launcher checks. Accelerator jobs are submitted
  by the user; do not start training or inference without an explicit request.
- Keep keys, environments, weights, run outputs, and private reports out of Git.
- Read `docs/platform/` contracts before simulator changes; read `bench_env/docs/task/`
  contracts before task authoring. Regenerate navigation artifacts after route changes.
- After changes, run focused regression tests and `python scripts/check_release.py`.
  For broad TypeScript changes, run `npx tsc --noEmit` as well as the frontend build.
