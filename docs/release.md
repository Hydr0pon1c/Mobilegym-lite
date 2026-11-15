# Source release

The source distribution includes the simulator, task definitions and snapshots,
feature masks, fixed task splits, Full/feature0/MobileGym-Lite configurations,
and training, export, and evaluation tools. Download companion media separately
as described in the [README](../README.md#start-the-simulator).

## Verify and package

From the repository root:

```bash
python scripts/check_release.py
# After intentional source changes, refresh the inventory:
python scripts/check_release.py --write-manifest
# Create a source ZIP without Git history, dependencies, or local run outputs:
python scripts/check_release.py --zip ../Mobilegym-lite-release.zip
```

[release-manifest.json](../release-manifest.json) records SHA-256 hashes of the
released files. The ZIP contains only that verified inventory and the manifest;
it can be checked after extraction with the same command, without Git.

Keep credentials in environment variables or an ignored local `.env` file.
Use `MODEL_PATH`, `MOBILEGYM_DATA_ROOT`, and optionally `RL_VENV` for machine-local
paths. Do not include weights, run outputs, notebook execution results, or private
reports in the source inventory. The checker scans for common credential formats
and personal filesystem paths; review any additional material before publishing.

## Reproducibility and licenses

Use the fixed task lists, seeds, and configurations in the [training guide](../mobilegym-rl/README.md).
Keep the model, frontend build, runtime, and hardware settings consistent across
environments. The accelerator package versions are listed in that guide.

Retain `LICENSE`, `LICENSE-DATA`, `NOTICE`, `DISCLAIMER.md`,
`THIRD_PARTY_NOTICES.md`, and vendored licenses. Original attribution and the
file scope in [copyright-scope.json](../copyright-scope.json) remain included.
Code, bundled app data, and external companion media have separate license terms.
