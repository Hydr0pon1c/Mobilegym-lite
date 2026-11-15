# MobileGym-Lite task split

The released Full, feature0, and MobileGym-Lite configurations use the fixed
`env_effect_v3` template lists. This split was calibrated for Qwen3-VL-4B-Instruct;
it is separate from the upstream `train` and `test` benchmark splits.
The checked-in memberships are the input to reproduction and should stay fixed
when comparing environments.

| Set | Templates | Use |
| --- | ---: | --- |
| `env_effect_v3_train` | 192 | Training |
| `env_effect_v3_test` | 155 | Combined evaluation |
| `env_effect_v3_shared` | 132 | Templates appearing in both train and test |
| `env_effect_v3_train_only` | 60 | Training-only templates |
| `env_effect_v3_ood_test` | 23 | Evaluation templates absent from training |
| `env_effect_v3_core` | 215 | Union of train and test |
| `env_effect_v3_stress` | 106 | Separate L4 task pool |
| `env_effect_v3_excluded` | 95 | Other upstream templates outside the core |

Train and test intentionally share template IDs. Different sampling seeds
produce the concrete task instances. Use one instance per template, training
seed 41, and evaluation seeds 42, 43, 44. Three evaluation seeds produce 465
episodes; they are not independent training repeats. The default training batch
size is 12, giving 16 outer steps per epoch.

Standard evaluation uses the Full environment. Report the combined test score
and, when needed, its shared and template-OOD components separately.

```bash
python -m bench_env.run --list --split env_effect_v3_train
python -m bench_env.run --list --split env_effect_v3_test
python -m pytest bench_env/tests/common/test_env_effect_v3_split.py -q
```

`env_effect_v3_candidates.csv` maps each of the 416 upstream templates to its
fixed partition. `env_effect_v3_manifest.json` records list sizes, relationships,
and sampling settings. Training commands are in the
[training guide](../../mobilegym-rl/README.md).
