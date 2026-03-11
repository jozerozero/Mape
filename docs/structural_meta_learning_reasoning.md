# Structural Meta-Learning for Tabular Foundation Models

## 0. 一句话定义
把“推理”从静态前向映射，升级为**结构探测 + 局部动力学搜索（TTT式快速适配）+ 前向映射**。

---

## 1. Motivation：为什么现在的 Tabular Foundation Model 还不够

当前多数 Tabular Foundation Model（TFM）在训练时吸收了大量“表内统计模式”，但**跨表迁移（table-to-table transfer）**仍然有限，尤其在以下场景：

1. **结构分布漂移（Structural Shift）**：
   新表与训练表在因果结构上不同（例如由 Fork 主导转为 Collider 主导），即使边际分布看起来相近，最优决策边界也会变化。
2. **机制切换（Mechanism Shift）**：
   $P(Y|X)$ 不仅数值变化，连依赖关系类型都变化（条件独立关系重排）。
3. **小样本目标表（few-batch adaptation）**：
   真实部署中通常只拿到新表很少批次，无法全量微调大模型。

### 缺乏迁移的后果

1. 新表性能波动大（高方差）
2. 易过拟合表面相关而非稳定机制（spurious correlations）
3. 推理阶段对非平稳环境反应慢
4. 每次换表都需昂贵重训/长时微调，部署成本高

---

## 1.1 为了实现迁移，我们的核心 Insight

### Insight A：迁移对象不应是“固定函数”，而应是“可快速收缩的参数态”

模型应学习一个初始化点 $\phi$，使其在新表上通过 1-2 步自监督更新即可靠近该表的结构最优点：

$$
\min_{\phi} \sum_{\mathcal{S}_i \sim \mathcal{P}(\text{SCM})}
\mathcal{L}_{\text{test}}\Big(\theta_{\text{frozen}},\ \phi'_i\Big),\quad
\phi'_i = \phi - \eta\nabla_\phi \mathcal{L}_{\text{self}}(\mathcal{D}^{\text{adapt}}_i;\theta_{\text{frozen}},\phi)
$$

这就是“微调后误差最小化”的元学习目标（MAML/TTT思想融合）。

### Insight B：结构原语可组合（Chain / Fork / Collider）

尽管具体 SCM 千差万别，但很多可迁移信息可以压缩到少量因果原语响应器（primitive experts）。

### Insight C：推理时应先做“结构对齐”，再做“数值预测”

先估计当前批次偏向哪类结构，再激活对应参数组合并小步适配，最后预测。

---

## 1.2 如何实现这个 Insight（详细可执行设计）

下面给出一个与你想法一致、能直接工程化的方案。

### 1.2.1 模型分解：冻结主干 + 结构敏感插件

1. 冻结 backbone：$f_{\theta}$（例如现有 TFM 主干）。
2. 在中间层插入轻量插件 $A_\phi$（Adapter/LoRA 形式都可）：
   $$h' = h + A_\phi(h)$$
3. 插件参数拆为三组原语子参数：
   - $\phi_{\text{chain}}$
   - $\phi_{\text{fork}}$
   - $\phi_{\text{collider}}$

### 1.2.2 结构探测头（Structural Probing Head）

输入当前 batch 的统计指纹 $s(D)$，输出原语权重 $\alpha \in \Delta^3$：

$$
\alpha = \text{softmax}(g_\psi(s(D))) = [\alpha_c,\alpha_f,\alpha_{col}]
$$

建议的 $s(D)$（可逐步加复杂度）：

1. 二阶统计：协方差、相关系数谱
2. 条件相关近似：partial correlation
3. 信息论特征：离散化互信息摘要
4. 损失地形特征：小批 Hessian trace / gradient variance

### 1.2.3 动态组合插件参数

$$
\phi_{\text{active}} = \alpha_c\phi_{\text{chain}} + \alpha_f\phi_{\text{fork}} + \alpha_{col}\phi_{\text{collider}}
$$

用 $\phi_{\text{active}}$ 参与当前批次前向。

### 1.2.4 推理期 TTT（局部动力学搜索）

在无标签或弱标签场景，使用自监督目标更新插件（1-2步）：

$$
\phi_{\text{active}}' = \phi_{\text{active}} - \eta\nabla_\phi \mathcal{L}_{\text{self}}(D; \theta, \phi_{\text{active}})
$$

可选 $\mathcal{L}_{\text{self}}$：

1. 特征掩码重建（mask-and-reconstruct）
2. 行列对比学习一致性
3. 对预测熵最小化（加稳定性约束）

### 1.2.5 元训练（外循环）

任务级训练流程（每个任务是一个 SCM 采样数据集）：

1. 采样任务 $\mathcal{T}_i$（含 chain/fork/collider 不同比例）
2. 用 $\mathcal{D}^{adapt}_i$ 做 1-2 步插件内循环更新
3. 用 $\mathcal{D}^{eval}_i$ 计算外循环损失并更新初始插件参数
4. 主干 $\theta$ 可冻结或半冻结（建议先冻结）

### 1.2.6 与你当前 data-selection 思路的统一

你在 `data_selection` 分支已经有“第一周期累计方向、后续按相似度重加权”的机制。
这个机制可直接作为 **TTT 内循环样本重权重器**：

1. 第 1 周期：估计参考梯度方向（建议只看 ICL/插件梯度）
2. 第 2 周期起：相似样本权重高，不相似样本权重低
3. 结果：TTT 更新更偏向“结构一致样本”，减少错误方向牵引

### 1.2.7 工程落点（Mape 项目）

可按以下模块切分实现，风险最小：

1. `adapter.py`：实现 Causal-Sensitive Adapter 与 3 组子参数
2. `probe.py`：实现结构探测头 $g_\psi$
3. `run.py`：新增 meta-train / ttt-adapt 两阶段流程
4. `train_config.py`：新增参数（`--meta_inner_steps`、`--adapter_rank`、`--probe_features` 等）
5. 保留现有 backbone 和主训练循环，先做 feature flag 控制（可回退）

---

## 1.3 实验验证方案（详细）

### 1.3.1 研究问题（RQ）

1. **RQ1**：结构性元学习是否提升跨表泛化（unseen SCM）？
2. **RQ2**：结构探测 + 原语加权是否优于统一插件？
3. **RQ3**：推理期 1-2 步 TTT 是否带来稳定收益？
4. **RQ4**：梯度方向 data selection 是否减少负迁移？

### 1.3.2 数据与划分

#### 合成任务族（主实验）

1. 基于 SCM 生成器控制结构原语比例：
   - train: 多任务混合（覆盖广）
   - val/test: OOD 结构组合（例如 collider 比例突增）
2. 控制难度轴：
   - 噪声强度
   - 观测维度
   - 隐变量/混杂比例

#### 真实表格任务（迁移实验）

1. OpenML 分类任务子集（按 schema 差异分层）
2. UCI + Kaggle 中可复现实验集
3. 若可用：TALENT/Benchmark 任务簇

### 1.3.3 对比基线（必须）

1. Frozen Backbone（无适配）
2. Full Fine-tuning（上限）
3. Adapter-only（无结构探测）
4. LoRA-only（无结构探测）
5. TTT-only（无元训练）
6. Meta-only（无推理时 TTT）
7. 你的完整方案（Meta + Structural Probe + Primitive Mixing + TTT + Data Selection）

### 1.3.4 指标

1. 主指标：AUC / Accuracy / NLL（分类）
2. 迁移指标：
   - OOD gap（ID 到 OOD 性能差）
   - Few-batch adaptation gain（适配前后提升）
3. 稳定性指标：
   - 不同随机种子标准差
   - 结构突变场景下 worst-case performance
4. 代价指标：
   - 额外参数量
   - 推理时延（含 1-2 步 TTT）
   - GPU memory 增量

### 1.3.5 消融实验（Ablation）

1. 去掉结构探测头（固定均匀权重）
2. 去掉原语拆分（单一插件）
3. 去掉元学习（仅常规训练）
4. 去掉 TTT（仅静态推理）
5. 去掉 data selection（等权更新）
6. 仅 ICL 梯度 vs 全参数梯度做 data selection
7. 内循环步数 0/1/2/3 对比

### 1.3.6 统计检验

1. 每个设置至少 5 seeds
2. 报告 mean ± std
3. 配对 t-test / Wilcoxon（按任务配对）
4. 报告效应量（Cohen’s d）

### 1.3.7 预期现象（可证伪）

1. 在结构 OOD 任务上，完整方案优于 adapter-only
2. 1-2 步 TTT 已可达到主要收益，>3 步边际收益下降
3. ICL-only 梯度 data selection 优于全参数梯度（噪声更低）

---

## 1.4 20 篇相关论文（按与你的方案相关性整理）

> 下面优先给“可直接指导设计/实验”的文献，并附核心关联点。

| # | 论文 | 年份/会议 | 与本方案的关系 | 链接 |
|---|---|---|---|---|
| 1 | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al.) | ICML 2017 | 元学习外循环基础（优化“适配后误差”） | https://arxiv.org/abs/1703.03400 |
| 2 | On First-Order Meta-Learning Algorithms / Reptile (Nichol et al.) | 2018 | 低成本近似 MAML，可作工程替代 | https://arxiv.org/abs/1803.02999 |
| 3 | Meta-SGD: Learning to Learn Quickly for Few-Shot Learning (Li et al.) | 2017 | 学习 per-parameter update rate，适合插件快速收缩 | https://arxiv.org/abs/1707.09835 |
| 4 | Meta-Learning with Implicit Gradients (iMAML) (Rajeswaran et al.) | NeurIPS 2019 | 稳定元优化，适用于大模型+小插件场景 | https://arxiv.org/abs/1909.04630 |
| 5 | Rapid Learning or Feature Reuse? (ANIL) (Raghu et al.) | ICLR 2020 | 证明“只适配头部/局部模块”可行，支持冻结主干思路 | https://arxiv.org/abs/1909.09157 |
| 6 | Test-Time Training with Self-Supervision (Sun et al.) | ICML 2020 | TTT核心范式：推理时小步自监督更新 | https://arxiv.org/abs/1909.13231 |
| 7 | Tent: Fully Test-Time Adaptation by Entropy Minimization (Wang et al.) | ICLR 2021 | 纯测试时适配经典基线 | https://arxiv.org/abs/2006.10726 |
| 8 | CoTTA: Continual Test-Time Domain Adaptation (Wang et al.) | CVPR 2022 | 连续测试流适配，与你的在线推理场景相关 | https://arxiv.org/abs/2203.13591 |
| 9 | EATA: Efficient Test-Time Model Adaptation without Forgetting (Niu et al.) | 2022 | 样本筛选/防遗忘机制可借鉴到 data selection | https://arxiv.org/abs/2204.02610 |
| 10 | TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second (Hollmann et al.) | 2022 | 表格基础模型代表作，适合作为对比或先验 | https://arxiv.org/abs/2207.01848 |
| 11 | Revisiting Deep Learning Models for Tabular Data (Gorishniy et al.) | NeurIPS 2021 | FT-Transformer 强基线与训练细节参考 | https://arxiv.org/abs/2106.11959 |
| 12 | TabTransformer: Tabular Data Modeling Using Contextual Embeddings (Huang et al.) | 2020 | 表格 Transformer 结构化编码参考 | https://arxiv.org/abs/2012.06678 |
| 13 | SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training (Somepalli et al.) | 2021 | 行/列建模与自监督目标参考 | https://arxiv.org/abs/2106.01342 |
| 14 | TabLLM: Few-shot Classification of Tabular Data with LLMs (Hegselmann et al.) | 2023 | 跨任务 few-shot 迁移视角，补充对照 | https://arxiv.org/abs/2210.10723 |
| 15 | TransTab: Learning Transferable Tabular Transformers Across Tables (Wang et al.) | 2022 | 直接针对跨表迁移，与你的 motivation 强相关 | https://arxiv.org/abs/2205.09328 |
| 16 | Causal Inference using Invariant Prediction (Peters et al.) | JRSS-B 2016 | “跨环境不变机制”理论基础 | https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12167 |
| 17 | Invariant Risk Minimization (Arjovsky et al.) | 2019 | 学习跨环境稳定机制的代表思想 | https://arxiv.org/abs/1907.02893 |
| 18 | Toward Causal Representation Learning (Schölkopf et al.) | Proc. IEEE 2021 | 因果表示学习全景，支持结构原语建模动机 | https://arxiv.org/abs/2102.11107 |
| 19 | Parameter-Efficient Transfer Learning for NLP (Houlsby et al.) | ICML 2019 | Adapter 插件范式来源，适配你“冻结主干”需求 | https://arxiv.org/abs/1902.00751 |
| 20 | LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.) | 2021 | 轻量插件实现首选之一，可直接迁移到 TFM | https://arxiv.org/abs/2106.09685 |

---

## 可直接执行的下一步（建议）

1. **先做最小可行版本（2周）**：冻结主干 + 单一 Adapter + TTT（无原语拆分）
2. **再加结构探测（2周）**：引入 $\alpha$ 加权三原语参数
3. **最后上元学习（2-4周）**：外循环优化“适配后误差”
4. **并行做你已有 data-selection 机制对接**：优先验证“ICL-only 梯度是否更稳”

这个路线能把风险拆开，保证每一步都可验证。
