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

---

## 2. 补充设计 A：基于 SCM 生成器控制结构原语（Chain/Fork/Collider）

本节回答两个工程问题：
1. 原语可控的 SCM 生成器怎么设计？
2. 这种结构信息以什么形式保存，便于训练和统计？

### 2.1 设计目标（先定义“可控”）

我们希望每个合成任务可以通过参数显式控制：

1. 节点规模：`n_obs`（观测节点数，含 `X` 与 `Y`）
2. 目标节点：`y_idx`（建议固定，便于跨任务统计）
3. 原语混合比例：`motif_mix = {chain, fork, collider}`
4. 背景稀疏度：`bg_edge_rate`
5. 混杂强度：`latent_confounder_rate`（决定双向边出现概率）
6. 图约束：`max_in_degree`, `max_out_degree`, `acyclic=True`
7. 可选目标统计：`target_ratios = {parent, child, sibling, other}`

### 2.2 参考 motif 工作的思路：Backbone + Motif Injection

图神经网络中常见的 motif 合成套路是：
1. 先造一个背景图（backbone）
2. 再按分布注入 motifs
3. 最后做结构标签和任务标签生成

可参考的实践风格包括：
1. motif-specific 邻接通道（不同 motif 不同通道）
2. BA/ER 背景图上注入 house/cycle 等结构块
3. motif 注入后再做结构解释/识别任务

对 SCM 场景可直接改写为：
1. 先生成背景 DAG
2. 按 `motif_mix` 注入 chain/fork/collider 模板
3. 再注入 latent confounder（双向边）
4. 在该图上采样结构方程并生成数据

### 2.3 生成流程（可执行版本）

#### Step 1：生成背景 DAG（不含双向边）

1. 对节点采样拓扑顺序 `pi`
2. 仅允许从前到后连边，确保无环
3. 按 `bg_edge_rate` 采样边
4. 若超出 `max_in_degree/max_out_degree`，做局部重采样

得到 `A_dir_bg in {0,1}^{N x N}`。

#### Step 2：按比例注入原语模板

令总注入数量为 `K_motif`，按 `motif_mix` 分配：
1. Chain 模板：`a -> b -> c`
2. Fork 模板：`a <- b -> c`
3. Collider 模板：`a -> b <- c`

实现要点：
1. 优先从“可用节点池”抽样，避免与既有边冲突
2. 必要时允许覆盖背景边（并记录 provenance）
3. 可配置是否“锚定 Y”：
   - `collider@Y`: `x_i -> y <- x_j`
   - `fork@Y`: `y -> x_i, y -> x_j`
   - `chain-through-Y`: `x_i -> y -> x_j` 或 `x_i -> x_j -> y`

#### Step 3：注入 latent confounder（双向边）

对节点对 `(i,j)` 按 `latent_confounder_rate` 采样隐混杂：
1. 若命中，记录 `i <-> j`
2. 双向边不写入 DAG 拓扑通道，单独存入 `A_bidir`

这样可同时满足：
1. 有向结构可用于拓扑生成
2. 隐混杂可用于“兄弟关系”统计和结构偏移控制

#### Step 4：结构修正（可选，保证比例达标）

如果需要逼近目标比例（如真实数据中的父/子/兄弟/其他）：
1. 计算当前 `Y` 邻域统计
2. 与 `target_ratios` 比较
3. 迭代执行局部 rewiring：
   - 补 parent：增加 `x -> y`
   - 补 child：增加 `y -> x`
   - 补 sibling：增加 `x <-> y`（通过 latent confounder 标注）
   - 降 other：优先在 Y 邻域外删边
4. 直到误差小于容差或迭代上限

#### Step 5：在固定图上采样机制

图确定后再采样每个节点机制（MLP/Tree）：
1. 按拓扑序生成有向部分
2. 对双向边以“共享噪声项/隐变量注入”模拟
3. 输出 `(X, y)` 与结构标签

核心原则：**结构与机制解耦**。  
这样你可在同一结构下替换不同函数族，单独研究“结构迁移 vs 函数迁移”。

### 2.4 与当前 Mape 代码的衔接建议（不改代码版）

你当前 `MLPSCM/TreeSCM` 的 `graph_sparsity` 更像“相关性削弱”，不是显式 motif 控图。  
建议下一步新增一个“结构先行”层：

1. `SCMGraphGenerator`：只负责生成 `A_dir/A_bidir/motif_meta`
2. `SCMMechanismSampler`：在给定图上生成数据
3. `SCMTaskBuilder`：组装成训练任务并附带 graph_pack

这样不会破坏现有生成器逻辑，可以 feature flag 逐步切换。

### 2.5 推荐的参数接口（草案）

```text
--scm_graph_mode motif_controlled
--n_obs 20
--y_idx -1
--bg_edge_rate 0.08
--motif_total 12
--motif_mix_chain 0.30
--motif_mix_fork 0.45
--motif_mix_collider 0.25
--latent_confounder_rate 0.10
--max_in_degree 4
--max_out_degree 4
--target_parent_ratio 0.08
--target_child_ratio 0.17
--target_sibling_ratio 0.03
--target_ratio_tolerance 0.02
```

### 2.6 质量控制（必须有）

每次生成后建议记录并检查：
1. DAG 无环性（仅 `A_dir`）
2. 度分布与稀疏度
3. `Y` 的 parent/child/sibling/other 比例
4. motif 计数偏差（目标 vs 实际）
5. 数据是否有 NaN/Inf

---

## 3. 补充设计 B：结构以什么形式保存

建议分两层保存：训练时张量格式 + 落盘格式。

### 3.1 训练时（dataloader）统一返回 `graph_pack`

对 batch 大小 `B`、padding 后节点数 `N_max`：

1. `adj_dir`: `bool/uint8`, shape `(B, N_max, N_max)`  
   表示有向边 `i -> j`
2. `adj_bidir`: `bool/uint8`, shape `(B, N_max, N_max)`  
   表示双向边 `i <-> j`（隐混杂）
3. `node_mask`: `bool`, shape `(B, N_max)`  
   有效节点 mask（处理不同维度数据）
4. `y_index`: `int64`, shape `(B,)`  
   每个样本里 Y 的节点索引
5. `motif_type`（可选）: `uint8`, shape `(B, N_max, N_max)`  
   边来源标签：`0=bg,1=chain,2=fork,3=collider,4=latent`

这是最直接支持你“先不喂模型，只做统计和对齐”的格式。

### 3.2 落盘格式（推荐）

每个 batch 一份 `.npz` 或 `.pt`：

1. 原有数据：`X, y, d, seq_len, train_size`
2. 图结构：`adj_dir, adj_bidir, node_mask, y_index`
3. 元信息：`meta.json`
   - 生成参数（motif_mix、稀疏度、confounder_rate）
   - 随机种子
   - 当前 batch 的结构统计摘要

### 3.3 稀疏存储（大维度时）

当 `N_max` 大而图稀疏，建议 edge-list：

1. `edge_index_dir`: `int32[2, E_dir]`
2. `edge_index_bidir`: `int32[2, E_bidir]`
3. `edge_type`: `uint8[E_total]`
4. `graph_ptr`: `int32[B+1]`（批次拼接边界）

优点：
1. 节省磁盘与内存
2. 便于图算子库（PyG/DGL）直接消费

### 3.4 建议的“统计审计表”

额外维护一个 `summary.tsv/parquet`（每图一行）：

1. `graph_id, n_obs, sparsity`
2. `parent_ratio, child_ratio, sibling_ratio, other_ratio`
3. `chain_count, fork_count, collider_count, latent_count`
4. `seed, split(train/val/test), config_hash`

这张表对你后续做 data selection 和结构分桶评估非常关键。

---

## 4. 一个最小可行实现顺序（仅设计）

1. 先实现 `A_dir/A_bidir + y_index + node_mask` 保存与统计（不改训练）
2. 再引入 motif injection（先不做 target ratio 修正）
3. 再加 target ratio 修正器（保证生成分布贴近真实）
4. 最后将 `graph_pack` 接入结构探测头与推理权重模块

这样可以确保每一步都可观测、可回滚。
