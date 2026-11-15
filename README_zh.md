# MobileGym-Lite

[English](README.md)

MobileGym-Lite 是用于 GUI 智能体训练与评测的浏览器移动环境，包含 MobileGym
模拟器、基于状态的任务评判，以及 rLLM/VERL GSPO 训练链路。功能裁剪在每个
任务回合内生效，保留任务数据、必需功能和原始评判逻辑。

| 环境 | 配置文件 | 功能模式 |
| --- | --- | --- |
| Full | [full.json](configs/training/full.json) | `null` |
| feature0 | [feature0.json](configs/training/feature0.json) | `feature0` |
| MobileGym-Lite | [feature0-selective.json](configs/training/feature0-selective.json) | `feature0_selective_v1` |

三组配置均在 Full 环境评测。MobileGym-Lite 对应 `feature0_selective_v1`，
与内部兼容模式 `lite_v1` 不同。具体规则见[功能裁剪说明](docs/feature-pruning.md)。

## 安装与启动

需要 Node.js 22 或更新版本和 Python 3.12。在仓库根目录执行：

```bash
npm ci
python -m venv .venv
source .venv/bin/activate
python -m pip install -r bench_env/requirements.txt
python -m playwright install chromium
curl -fL -o mobilegym-data.tar.gz \
  https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz
tar -xzf mobilegym-data.tar.gz
npm run dev -- --host 127.0.0.1
```

确认配套数据解压为仓库根目录下的 `mobilegym-data/`，然后访问
`http://127.0.0.1:3000`。可选配置见 [.env.example](.env.example)。
智能体模型的地址和密钥通过 `--model-base-url`、`--model-api-key` 单独设置。

## 复现入口

固定划分包含 192 个训练模板和 155 个评测模板，每个模板采样一个实例；
训练种子为 41，评测种子为 42、43、44。任务列表与采样约定见
[划分说明](bench_env/splits/ENV_EFFECT_V3.md)。

[训练指南](mobilegym-rl/README.md)提供硬件依赖、训练、恢复、模型导出与评测命令。
以下命令仅检查 Lite 配置，不启动服务或训练：

```bash
MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct \
MOBILEGYM_CONFIG_ONLY=1 RUN_NAME=lite_check \
TRAIN_CONFIG="$PWD/configs/training/feature0-selective.json" \
bash mobilegym-rl/cookbooks/mobilegym/run_qwen3_vl_4b_env_effect_v3_full.sh
```

用 `MODEL_PATH` 指定本地模型，`MOBILEGYM_DATA_ROOT` 指定输出目录。
训练入口默认使用 16 个设备，运行前按实际硬件设置 `CUDA_VISIBLE_DEVICES`。
比较三种环境时，应保持模型、前端构建、运行时、任务采样及奖励设置一致。

## 检查与发布

```bash
python scripts/check_release.py
npm test
npm run build
python -m pytest bench_env/tests/common/test_feature_completeness.py \
  bench_env/tests/common/test_completeness.py \
  bench_env/tests/common/test_env_effect_v3_split.py -q -m 'not live'
```

更多内容见[文档索引](docs/README.md)、[任务运行指南](bench_env/README.md)
和[源码 ZIP 打包说明](docs/release.md)。

本项目基于 MobileGym、rLLM 和 VERL。代码使用 Apache-2.0；应用数据及配套资源
适用 CC BY-NC 4.0 和相应第三方条款。分发时保留
[LICENSE](LICENSE)、[LICENSE-DATA](LICENSE-DATA)、[NOTICE](NOTICE)、
[DISCLAIMER](DISCLAIMER.md)及[第三方声明](THIRD_PARTY_NOTICES.md)。
