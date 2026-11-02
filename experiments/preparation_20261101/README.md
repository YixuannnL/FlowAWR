# FlowAWR H20 实验准备（2026-11-01）

当前状态：用户已确认，A5.1 已正式提交，平台 Queuing，尚未分配Pod/GPU；A6.1 未提交。任务及输出路径见 `a5_1_submission.json`。以下保留前置验证记录；正式运行使用确认时的固定代码包，后续实验登记修改不改变已提交代码快照。

| 编号 | 实验 | 启动器 | 状态 |
| --- | --- | --- | --- |
| A5.1 | A1 + max_grad_norm=0.2 | scripts/launchers/run_a5_1_flowawr_sd3_pickscore_gradclip_0p2_8gpu.sh | 已提交，排队中 |
| A5.2 | A1 + max_grad_norm=0.1 | 后续创建 | 预留编号 |
| A5.3 | A1 + max_grad_norm=0.3 | 后续创建 | 预留编号 |
| A6.1 | frozen-current two-update + PickScore | scripts/launchers/run_flowawr_sd3_two_update_8gpu.sh | 已登记，待确认 |

A5.1 复制 A1 启动器，训练入口仍为 `scripts/train_awr_sd3.py`，配置仍为 `config/awr.py:sd3_pickscore`。唯一训练超参数变化是 `--config.train.max_grad_norm=0.2`。原 A1 启动器和训练代码未改。新脚本额外适配 H20 的资产路径、独立 Python 环境、离线检查及 CPFS 输出。

A6.1 登记已有 two-update 训练实现，不叠加 A5 的裁剪修改：max_grad_norm=1.0；每个 rollout 两轮完整 batch 更新尝试，冻结首次更新前 current anchor、target 和归一化分母，复用样本/时间步/噪声；EMA=False，train.beta=0，decay_type=0。默认 REWARD=pickscore，其他 reward 尚未准备资产。

共同配置：8×H20，512×512，LoRA r32/alpha64，seed42，fp16，lr3e-4；每卡 batch9、rollout batches16、梯度累积16；48组×24图=1152图/rollout；采样/评估10/40步，guidance1，DPM2 deterministic，temperature1，energy[-3,3]。从头训练，不恢复 checkpoint；100000 epochs；每10 epoch评估/保存。A5.1 保留 EMA=True、KL=1e-4、一轮更新。

## 固定资产

- SD3.5 Medium 公共目录：`/cpfs/public_models/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80`
- PickScore 公共目录：`/cpfs/public_models/huggingface/hub/models--yuvalkirstain--PickScore_v1/snapshots/a4e4367c6dfa7288a00c550414478f865b875800`
- Processor 公共目录：`/cpfs/public_models/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c`
- `REWARD_CKPTS_DIR=/cpfs/yuxuanyao/models/flowawr/reward_ckpts_20261101`，其 `pickscore/model`、`pickscore/processor` 分别链接到上述固定版本。公共权重已经齐备，无需下载重复权重。
- 运行时归档：`/cpfs/yuxuanyao/models/flowawr/runtime_20261101/runtime.tar.gz`，SHA256=`6b46bfe8e0d7a36ecf4faa4888b11e5accb513691e8b587dc7f94d036dfca753`；同目录有 `SHA256SUMS` 和两份依赖 lock。这是部署归档，运行环境解压在 `/local-ssd`，没有在 CPFS 安装环境。
- 镜像：`600627331169.dkr.ecr.us-west-2.amazonaws.com/arcwm/train-aws@sha256:f8855ec9bf9629d19ac2536d64a351f4bdc9dc58743fc788c6c734a45864169b`
- Python3.12.3；torch2.6.0+cu124；torchvision0.21.0；transformers4.40.0；diffusers0.33.1；peft0.10.0。SwanLab0.9.0及protobuf6在 `/local-ssd/flowawr-swanlab`，与主环境 wandb0.18.7/protobuf5 隔离。

## 验证与限制

已通过：shell语法、CSV结构及原14行不变、公共模型组件和shard存在、safetensors文件大小与header一致、离线processor加载、两个训练入口导入、A1配置核验（1.0/9/16/16）、CPU实际PickScore单图打分（原始分数约20.7335）、两个启动器的最终八卡参数预览（torchrun替换成只打印参数的程序）、运行时归档SHA256校验及从归档重建环境后的离线检查。

未验证：GPU CUDA/NCCL、八卡显存、反向和optimizer更新、checkpoint恢复。当前 CPU 验证不能证明正式八卡训练成功。历史 A1 使用过不同 GPU/Python/PyTorch 环境，本次仅保证同一代码配置上的参数对照，不保证与历史运行逐步一致。

## 确认后启动的入口

先发布最终固定代码包，并核对提交计划。发布与提交预览的回执保存在本地 `.cache/preparation_20261101/`（不打进代码包）。本次准备不会执行真实 GPU submit。

A5.1 前台命令：

```bash
bash scripts/launchers/prepare_flowawr_h20_runtime.sh
exec bash scripts/launchers/run_a5_1_flowawr_sd3_pickscore_gradclip_0p2_8gpu.sh
```

A6.1 前台命令（仅登记，不自动提交）：

```bash
bash scripts/launchers/prepare_flowawr_h20_runtime.sh
exec bash scripts/launchers/run_flowawr_sd3_two_update_8gpu.sh
```

两个实验独立平台 run，各写入其 `$KOALA_RUN_DIR/logs`，含日志、逐样本数据和checkpoint。默认SwanLab离线。训练时长沿用100000 epochs，平台 Normal 最长168h，到期后的自动续训未配置。

CPU准备任务：`yuxuanyao-flowawr-prep-fffd0e5a7b-debug-20261101-183254`，run-id=`bcd19e949a5f4353a2f403c0be0155b0`；不占GPU，72h租期。
