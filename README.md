# 本次四个 FlowAWR 实验结果

源代码版本：`c3beaf68c69d24c38391801e610a4400bece1c14`（`YixuannnL/FlowAWR`）。
这是一份独立的结果快照，范围为本次 A1、P1、P2、A2.1 四个实验。

A1 已更新为 2026-10-24 的采集结果；P1、P2、A2.1 保留 2026-10-23 的采集结果。
下列状态均指各自采集时刻，不代表当前实时任务状态。

| 实验 | 内容 | 快照状态 | 进度 |
| --- | --- | --- | --- |
| A1 | SD3.5-Medium / PickScore，τ=1.0 基线 | 仍在训练 | train reward step 2125；eval step 2120 |
| P1 | checkpoint-900 梯度 probe | 完成 | 2 轮，2,304 个样本 |
| P2 | step-0 / base model 梯度 probe | 完成 | 2 轮，2,304 个样本 |
| A2.1 | SD3.5-Medium / PickScore，τ=0.3 | 仍在训练 | train reward step 649；eval step 640 |

每组均为 2 节点、16 卡。训练每轮 `16 × 9 × 8 = 1152` 张图，
48 个 prompt groups、每组 24 张，配置每轮一次参数更新。
两个 probe 均使用 `num_rounds=2`、`sample_epoch=900`，不执行 optimizer update。

初次数据读取区间：`2026-10-23T02:50:23.303999+00:00` 至 `2026-10-23T02:52:13.603731+00:00`。
A1 本次更新的读取区间：`2026-10-24T10:12:31.388809+00:00` 至 `2026-10-24T10:16:32.787805+00:00`
（北京时间 10 月 24 日 18:12–18:16）。来源与转换核验见 `data/a1/SNAPSHOT.json`。
训练文件逐个固定前缀，并校验读取前缀未变化；完整来源与每个文件的时间、大小、
SHA-256 见 `SOURCE_MANIFEST.tsv`。导出期间训练继续运行。

## 训练指标

两组最近共同的评估 step 为 640，下表训练与评估使用同一 step：

| 实验 | step | train PickScore | eval PickScore |
| --- | ---: | ---: | ---: |
| A1 (τ=1.0) | 640 | 0.91273183 | 0.91550398 |
| A2.1 (τ=0.3) | 640 | 0.68195653 | 0.62024671 |

A1 此份快照的最新评估为 step 2120，PickScore=0.92521167。
分数保持训练程序原始标度。

更新审计：A1 各 rank 已记录 2125–2126 次更新尝试和实际更新，未见 AMP 跳步；
逐文件采集边界跨越一次更新，因此 rank 之间相差 1 次。A2.1 已记录 650 次尝试、
647 次实际更新。A2.1 在 **global step 167、169、177** 出现 3 次 AMP 跳步，
16 个 rank 的跳步记录一致。原程序即使 AMP 跳步仍递增 `global_step`，因此它不是
实际参数更新次数。完整计数见各自 `data/*/update_audit.json`。

两组训练原生记录格式是 **离线 SwanLab 0.9.0**。本快照读取其标量记录，验证每条
LevelDB CRC，并转换为 TensorBoard 的 `scalars` 插件、`DT_DOUBLE` 数据；
不改变 tag、step 或数值，不额外做平均或下采样。每组保留 94 个原始标量 tag，
包括训练指标和原始系统监控；共 1,287,242 个标量点（A1 为 985,337 点）。
原始秒／纳秒时间戳另存于 `scalars.jsonl.gz`，避免时间戳格式转换带来的精度损失。
SwanLab 的媒体和普通终端日志不进入结果分支。

## 查看四组结果

```bash
git clone --depth 1 --single-branch --branch tensorboard-results \
  git@github.com:YixuannnL/FlowAWR.git FlowAWR-results
cd FlowAWR-results
sha256sum -c SHA256SUMS
python3 -m pip install tensorboard
python3 -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

| 实验 | TensorBoard 目录 | 数据目录 |
| --- | --- | --- |
| A1 baseline | `runs/pickscore/flowawr_sd3_pickscore__2026.10.22_11.35.49` | `data/a1` |
| P1 checkpoint-900 probe | `runs/probes/flowawr_probe_ckpt900` | `probe_runs/flowawr_probe_ckpt900` |
| P2 step-0 probe | `runs/probes/flowawr_probe_step0` | `probe_runs/flowawr_probe_step0` |
| A2.1 temperature=0.3 | `runs/pickscore/flowawr_sd3_pickscore_tau0.3_c3.33__2026.10.22_12.34.23` | `data/a21` |

训练目录下的 `scalars.jsonl.gz` 保留每条标量及其原始时间戳；`scalar_inventory.tsv`
列出 tag、点数和 step 范围，`config.json` 为此次实际配置。
TensorBoard 的页面采样／平滑属于显示行为，导出文件保留了全部标量。

## Probe 数据与 SNR 分析

每个 `probe_runs/flowawr_probe_*` 目录完整保留 16 个 `sketch_rankNN.npz`、
16 个 `prompts_rankNN.json`、`probe_meta.json` 和原校验文件，共 34 个原始文件，
逐字节与共享盘一致。每个 probe 有 2,304 个样本，96 个组，每组 24 个；
所有梯度均为有限值。重算的最小逐 rank 梯度重建余弦分别为
P1 `0.9999978474`、P2 `0.9999978772`。

`analysis/summary.csv` 和 `summary.json` 由仓库现有
`scripts/analyze_grad_snr.py` 生成，使用 300 次随机分组、seed=0、17 个默认权重方案。
`analysis/integrity.json` 记录脚本哈希、参数和校验结果。
`runs/probes/*` 把这些分析值展示为 **step 0 的静态汇总**，不代表训练曲线，
也不把 `sample_epoch=900` 当成训练 step。

在代码 checkout 中重算（`RESULTS_DIR` 指向本结果仓库）：

```bash
python scripts/analyze_grad_snr.py \
  --probe_dir "$RESULTS_DIR/probe_runs/flowawr_probe_ckpt900" --splits 300 --seed 0
python scripts/analyze_grad_snr.py \
  --probe_dir "$RESULTS_DIR/probe_runs/flowawr_probe_step0" --splits 300 --seed 0
```

P1 的输入为此前指定的官方 r01 `checkpoint-900/lora`，完整来源在
`EXPORT.json` 的 `checkpoint_900_input`；本分支不包含该旧实验的训练结果或模型权重。
P2 的空 LoRA 路径表示 fresh Gaussian LoRA（B=0，初始输出等于 base model）。

## 实验命令与来源

以下命令在每组的两台节点各运行一次，`NODE_RANK` 分别为 0 和 1；同组共享
`MASTER_ADDR` 和 `MASTER_PORT`。实际部署 launcher 为
`/mnt/local_storage/flowawr-four-20261022/run_training.sh`。

```bash
launch=(torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")

# A1
"${launch[@]}" scripts/train_awr_sd3.py --config=config/awr.py:sd3_pickscore \
  --config.sample.num_batches_per_epoch=8 --config.train.gradient_accumulation_steps=8 \
  --config.train.temperature=1.0 --config.run_name=flowawr_sd3_pickscore

# P1: CKPT900_LORA 为 EXPORT.json 记录的 checkpoint-900/lora
"${launch[@]}" scripts/probe_grad_snr_sd3.py --config=config/awr.py:sd3_pickscore \
  --lora_path="$CKPT900_LORA" --out_dir=probe_runs/flowawr_probe_ckpt900 \
  --num_rounds=2 --sample_epoch=900

# P2
"${launch[@]}" scripts/probe_grad_snr_sd3.py --config=config/awr.py:sd3_pickscore \
  --lora_path="" --out_dir=probe_runs/flowawr_probe_step0 --num_rounds=2 --sample_epoch=900

# A2.1
"${launch[@]}" scripts/train_awr_sd3.py --config=config/awr.py:sd3_pickscore \
  --config.sample.num_batches_per_epoch=8 --config.train.gradient_accumulation_steps=8 \
  --config.train.temperature=0.3 --config.run_name=flowawr_sd3_pickscore_tau0.3_c3.33
```

部署对 `flow_grpo/pickscore_logging.py` 加入共享盘目录创建的有限重试；补丁哈希、
每组源代码包与部署包哈希都在 `EXPORT.json`。本快照保留实际配置与来源说明，
不包含训练源码、Dockerfile、checkpoint、生成图片或普通终端日志。
失败的 A1 启动尝试没有混入成功 run 的数据。

`MANIFEST.tsv` 一行对应一个实验，更新次数的区间表示各 rank 的最小值和最大值；`SOURCE_MANIFEST.tsv` 登记固定读取前缀；
`SHA256SUMS` 校验所有发布文件。Probe sketch 共约 1.22 GB，是本分支的主要体积。
更大的训练逐样本原始数据继续保存在 `EXPORT.json` 指定共享盘的 `training/*/raw_data/`，
本次按既有发布流程导出训练标量。
