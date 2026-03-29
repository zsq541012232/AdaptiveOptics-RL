# AdaptiveOptics-RL

基于 **Stable-Baselines3** 的自适应光学（AO）强化学习训练仓库，支持：

- AO 环境（Sharpening / Centering / Darkhole）。
- A2C / SAC 训练。
- 图像观测下的 `ResNet + CBAM` 特征提取器。
- TensorBoard 训练记录。

---

## 1. 你关心的几个关键点（结论）

### 1) ResNet+CBAM 支持预训练权重、**增加 CBAM 深度**、输入尺寸对齐、并保证 actor 可学习更新
本仓库已支持：

- 主干网络仍可用 `resnet18 / resnet34 / resnet50 / resnet101`，但重点是可以通过 `cbam_depth` 增加每个 stage 的注意力模块深度（例如 `cbam_depth=2`）。
- 可通过 `use_pretrained_resnet=True` 加载 torchvision 的 ImageNet 预训练权重。
- 当环境图像尺寸和 ResNet 期望输入不一致时，可配置 `resnet_input_size=(224, 224)`，在特征提取器内部自动双线性 resize。
- SAC 下显式设置 `share_features_extractor=False`，确保 actor 与 critic 分别拥有特征提取器参数，actor 路径中的提取器会参与 actor 反向传播更新（只要不冻结参数）。

### 2) 训练时实时观察（render）
新增配置：

- `render_during_training: bool`（开关）
- `render_every_n_steps: int`（每多少步渲染一次）

示例：

```python
config.update({
    "render_during_training": True,
    "render_every_n_steps": 200,
})
```

### 3) SAC 训练 reward 是一条直线：算法问题还是代码问题？
可能两者都有，建议按下面顺序排查：

1. **先排代码/日志问题**
   - 确认记录的是 step reward 均值而不是固定标量。
   - 确认环境返回 `reward` 在不同状态下确实变化。
2. **再排环境动态**
   - 当前 `reward=strehl`，若初始扰动与动作尺度导致 strehl 变化很小，曲线会近似直线。
   - 可考虑 reward shaping（例如 `delta_strehl`、加入 DM 惩罚）。
3. **最后做算法调参**
   - 增大 `buffer_size`、增加 `total_timesteps`。
   - 调整 `learning_rate`、`batch_size`、`train_freq`、`gradient_steps`。
   - 对观测做归一化、动作范围缩放。

建议最小可行优化：

- `resnet34 + pretrained + CBAM(depth=2)`（已在示例配置）
- `buffer_size >= 200000`
- `total_timesteps >= 300000`
- 先关闭 render，提高吞吐，稳定后再开启。

### 4) 能否和传统 SPGD 结合？
可以，常见三种融合策略：

1. **Warm-start**：先用 SPGD 做粗调，再切 RL 精调。
2. **Residual RL**：动作 = `a_spgd + a_rl`，RL 只学残差修正。
3. **Imitation/Offline 初始化**：先用 SPGD 轨迹预训练策略，再在线 SAC 微调。

### 5) 你已有“图像预测泽尼克系数”的 ResNet+CBAM 模型能否复用？
可以，推荐两种方式：

1. **作为特征提取器初始化权重**：把 encoder 主干权重加载到 `ResNetFeatureExtractor`。
2. **作为辅助任务（多头）**：RL 主任务之外，加一个 Zernike 预测头做联合训练（有助于表征学习稳定）。

要点：

- 输入归一化方式要一致。
- 输出动作维度与当前 `num_modes` 保持匹配。
- 若 domain gap 大，建议先冻结前几层，再逐步解冻。

---

## 2. 快速开始

### 环境依赖

```bash
pip install -U stable-baselines3[extra] gymnasium torchvision tensorboard hcipy matplotlib
```

### 训练（SAC）

```bash
python SAC-train.py
```

### 训练（A2C）

```bash
python A2C-train.py
```

---

## 3. 关键配置说明（`training_pipeline.py`）

- `policy_type`: `CnnPolicy` 或 `MlpPolicy`
- `use_image_observation`: 是否用图像观测
- `resnet_backbone`: `resnet18/34/50/101`
- `use_cbam`: 是否启用 CBAM
- `cbam_depth`: 每个残差块后串联多少个 CBAM（>=1）
- `use_pretrained_resnet`: 是否加载预训练权重
- `resnet_input_size`: 输入对齐尺寸，例如 `(224, 224)`
- `features_dim`: 投影后的特征维度
- `tensorboard_log_dir`: TensorBoard 日志目录（默认 `runs`）
- `render_during_training`: 训练时实时渲染开关
- `render_every_n_steps`: 渲染频率

---

## 4. 常见问题（FAQ）

### Q1: 为什么开了预训练但输入是 1 通道也能跑？
因为代码会自动把 ResNet 第一层 `conv1` 改成环境输入通道数，结构兼容。

### Q2: 为啥我开 render 后训练慢？
渲染会显著降低吞吐，建议只在 debug 阶段开启，或增大 `render_every_n_steps`。

### Q3: SAC reward 一直不涨怎么办？
优先检查：

- reward 计算是否有变化；
- 动作尺度是否过小/过大；
- 经验回放与学习率是否匹配。

然后再考虑 reward shaping 和 SPGD+RL 混合控制。

---

## 5. 项目结构

- `training_pipeline.py`：统一训练入口。
- `rl_feature_extractors.py`：`ResNetFeatureExtractor` 与 CBAM。
- `EnvironmentWrapper.py`：环境封装与观测格式。
- `SAC-train.py` / `A2C-train.py`：单次训练脚本。
- `SAC-experiment.py` / `A2C-experiment.py`：多次实验脚本。
