# FlowAWR：以当前预测构造冻结 target，进行两次更新

本文按以下思路分析：保留 FlowAWR 的逐样本 energy 和前向加噪训练，以**本轮第一次优化器更新前的当前模型**构造 target，再用同一份 target 更新两次。仅修改本文档，尚未修改训练代码或验证训练效果。

## 1. 结论

**这个方案在数学上合理：第一次获得逐样本奖励引导，第二次获得相同引导与朝初始预测的恢复力。** 必要条件是：target 停止梯度，且两次更新使用同一份冻结的 target。它可以作为 FlowAWR 中替代滞后策略 EMA 的实验方案，稳定性和效率需实测。

这里保留你的定义 `G = e * (x0 - x_current)`，不引入 ART 的跨样本协方差方向。另需明确：`x_current` 是模型对干净 latent 的重建，`x0` 是生成样本的干净 latent；二者都不是中间加噪状态 `xt`。

## 2. 严格定义与梯度

记一轮 rollout 后、第一次 optimizer step 前的参数为 `θ⁽⁰⁾`。对固定训练输入 `s=(xt,t,c)`：

\[
z_\theta(s)=x_t-t f_\theta(s),\qquad
z^{(0)}(s)=\operatorname{sg}[z_{\theta^{(0)}}(s)],
\]

其中 `fθ` 是 SD3 的速度输出，`zθ` 即代码中的 `x0_prediction`，`sg` 表示停止梯度。用本轮固定的 AWR energy `e` 定义：

\[
G_x(s)=e\,[x_0-z^{(0)}(s)],\qquad
T(s)=\operatorname{sg}[z^{(0)}(s)+G_x(s)].
\]

两次更新始终优化同一个目标：

\[
\ell_\theta(s)=\frac{1}{D w(s)}\|z_\theta(s)-T(s)\|^2,
\]

`D` 为 latent 元素数，`w>0` 为当前代码的归一化因子；为便于两轮严格对照，可在构造 target 时一并冻结 `w`。

**第一次更新前**，`zθ=z⁽⁰⁾`，所以重建空间中的输出下降方向为：

\[
-\nabla_z\ell\big|_{\theta^{(0)}}=\frac{2}{Dw}G_x.
\]

**第二次更新前**，令 `d_x=zθ⁽¹⁾-z⁽⁰⁾`，同一 target 给出：

\[
-\nabla_z\ell\big|_{\theta^{(1)}}=\frac{2}{Dw}(G_x-d_x).
\]

因此第二次更新包含奖励引导 `G_x` 和恢复项 `-d_x`，无需额外添加恢复 loss。这里的“方向”指输出空间梯度；实际参数更新还经过模型 Jacobian、梯度累积、裁剪及 AdamW，不能理解成每个样本的输出恰好移动 `G_x`。

**速度空间的量纲需要区分。** 你写的 `G=e*(x0-x_current)` 是重建空间位移。若写速度目标，对 `t>0` 应为：

\[
f_{\rm target}=f_{\theta^{(0)}}-G_x/t.
\]

第一次速度输出的下降方向为 `-t*G_x` 的正比例量，第二次为 `-t*G_x-t²*(fθ⁽¹⁾-fθ⁽⁰⁾)` 的正比例量。实现时直接保留 x0 重建损失，避免除以很小的 `t`。

## 3. 为什么必须冻结，以及能得到什么结论

- **不能让 target 参与求导。** 若直接写 `T=zθ+e*(x0-zθ)` 而不 detach，则残差为 `e*(zθ-x0)`，平方损失变成 `e²*||zθ-x0||²`，负 energy 的排斥方向丢失。
- **不能在第二次更新后重新锚定当前预测。** 即使 detach，每轮都重算 `T=zθ.detach()+e*(x0-zθ.detach())`，也只会持续产生当前残差方向，没有上述相对 `z⁽⁰⁾` 的恢复项。
- **“纯 reward guidance”成立于所定义的 policy loss。** `e>0` 拉向样本，`e<0` 推离样本，`e=0` 时第一次无引导、第二次仍可约束其他样本更新造成的漂移。因此不宜直接过滤所有零 energy 样本。
- **这不是稳定性的证明。** 该逐样本方向仍使用前向加噪残差，不能据此宣称消除了论文分析的前向/反向一致性问题，也不保证等于真实期望奖励的无偏梯度。借鉴的是 ART 的冻结锚点与第二次更新恢复机制。[论文 §3–4](https://arxiv.org/html/2610.12572v1#S4)

原代码的额外参考模型正则由 `config.train.beta` 控制。若保留它，总梯度还包括参考模型正则；要严格检验“第一轮只有上述奖励引导”，应设置 `config.train.beta=0`。这与顶层 `config.beta` 是不同参数。

## 4. 当前代码上的最小实现方案

按你对“每个样本两轮更新”的描述，采用：**固定同一批 `(x0, xt, t, e, T)`，第一遍累计一次 optimizer step，第二遍再累计一次 optimizer step。** 这与 ART 使用互斥子批的安排有所区别，但能直接检验你给出的逐样本推导。

### 4.1 更新前缓存输入与 target

修改 `scripts/train_awr_sd3.py` 的训练准备区（当前约 `:960`）及 energy loss 分支（约 `:1146`）：

1. 在任何 optimizer step 前，确定本轮训练使用的 timestep 与 noise，构造固定 `xt`。
2. 用此时的当前 `default` adapter，在 `no_grad` 下分微批计算 `z⁽⁰⁾`、`G_x`、`T`，以及冻结的 `weight_factor`。
3. 两遍训练复用上述缓存；第二遍不得重新采样 noise、重新选择 timestep 或重算 target。可打乱遍历顺序，但所有关联字段必须同步排列。

核心伪代码（`x0/xt/t/e` 均为固定训练数据）：

```python
# 在整批第一次 optimizer step 前完成；可分微批并缓存到 CPU
with torch.no_grad():
    f0 = predict_current(xt, t, cond)  # 当前 default；不是滞后的 rollout old
    z0 = xt - t_expanded * f0
    G = e_expanded * (x0 - z0)
    target_x0 = (z0 + G).detach()
    weight = (z0.double() - x0.double()).abs().mean(
        dim=tuple(range(1, x0.ndim)), keepdim=True
    ).clamp_min(1e-5).detach()

# 整批执行两遍，每遍仅在全部微批梯度累积后 step 一次
for update_index in range(2):
    optimizer.zero_grad()
    for mb in fixed_training_microbatches:
        prediction = predict_current(mb.xt, mb.t, mb.cond)
        z = mb.xt - mb.t_expanded * prediction
        per_item_loss = ((z - mb.target_x0)**2 / mb.weight).flatten(1).mean(1)
        # 沿用现有 DDP、AMP、累积归一化；全部微批完成后再裁剪和 step
        backward_accumulated(per_item_loss)
    optimizer_step_with_existing_amp_and_clipping()
```

当前 `num_inner_epochs` 循环（约 `:972`）每遍重新打乱 timestep，内层（约 `:1034`）又重新生成 noise，所以**只设置 `num_inner_epochs=2` 不足以实现此方案**。应先生成缓存，后让两遍循环回放它。两遍的模型模式与随机层设置也应一致，使第一次更新前的预测与缓存锚点一致。

### 4.2 优化器边界与配置

```python
config.train.energy_mode = True
config.train.num_inner_epochs = 2
config.train.gradient_accumulation_steps = num_batches  # 每遍整批只 step 一次
config.train.beta = 0.0                                 # 检验纯 policy guidance
config.train.ema = False                                # 若同时取消评估/保存 EMA
config.decay_type = 0                                   # rollout old 改为硬同步
```

`num_batches` 是脚本约 `:951` 计算的每遍训练微批数；反向累积次数仍乘 `num_train_timesteps`。标准 8 卡 preset 通常为 16 个微批，OCR 也应设为 16，避免一遍内已有两次 optimizer step。已有 `loss_mask` 若改变有效样本计数，需要保持归一化一致，并确认零 energy 样本未被无条件过滤。

**不要把每遍累积步数减半。** 本方案是整批更新两遍，合计两次 optimizer step；不是把整批分半后分别更新一次。每轮训练前向/反向工作量通常约为原单遍训练的两倍，另有缓存 target 的无梯度前向。

### 4.3 EMA 与 old adapter 如何处理

训练 target 不再读取滞后的 `old_prediction`。原 `old` adapter 可继续供采样使用，但若目标是替代训练相关 EMA，应将其 epoch 末更新（约 `:1302`）设为 `decay=0`，使下一轮 rollout 使用更新后的当前策略。它不再承担 loss 的滞后锚点角色。

最直接实现是缓存 `T` 后删除该新分支中不需要的 `old` 前向；也可复用 `old` 的存储，在两次更新前硬拷贝当前参数并整轮冻结，按需重算 `θ⁽⁰⁾` 的预测以减少缓存。后者只是当前模型快照的实现方式；不能沿用有 EMA 滞后的权重，也不能在两次更新之间刷新快照。

当 `old` 恰好等于 `θ⁽⁰⁾`，原 AWR target 与你定义的 target 在数值上可以相同；真正需要明确的是**锚点的来源、固定训练输入及两次实际更新的边界**。更换变量名本身不会改变算法。

评估/保存用的 `EMAModuleWrapper` 是独立机制；`train.ema=False` 只关闭这部分。现有 raw logger 使用 `pred_x0_old` 等字段，新分支应改为记录 `z0/G/d_x` 或暂时关闭 `log_raw_data`，避免沿用错误含义。

## 5. 最小验证

先在固定小批量上检查：target 无梯度；两次缓存内容完全相同；更新前 `zθ≈z⁽⁰⁾`；第一次输出梯度符合 `G_x`，第二次符合 `G_x-d_x`；负 energy 的方向正确；FP16 没有跳过 optimizer step。再与原 FlowAWR 对照，记录奖励、评估分数、梯度范数和第二次更新前的漂移，同时按生成样本数与 GPU 时间比较效率。

**最终判断：你的方案成立为“以当前预测构造冻结 AWR target，再进行两次更新”的方法。关键不是逐次用当前输出重建 target，而是只在第一次更新前构建一次，并在第二次更新时保留这个锚点。**
