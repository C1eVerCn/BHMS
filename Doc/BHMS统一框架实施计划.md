# BHMS 执行版方案：以 `xLSTM + Transformer + GraphRAG` 为主线的全生命周期预测与机理解释

更新日期：2026-03-22

## 0. 方案摘要

本轮执行方向固定为：

- 主干不变：项目继续围绕 `xLSTM + Transformer` 混合架构和 `GraphRAG` 展开，目标仍是“面向锂电池全生命周期预测与机理解释的 BHMS”
- `xLSTM` 是核心，不降级成纯 Transformer 路线：新版模型里 `xLSTM` 负责长程退化记忆和阶段连续性，`Transformer` 只负责全局依赖增强与跨阶段关系建模
- 当前优先级固定为：`仓库整理与基线提交 > 数据扩容 > xLSTM 生命周期模型 > GraphRAG 机理解释 > API/前端迁移 > benchmark 固化`

本文件是后续阶段更新的唯一主计划文档。后续每完成一个阶段，只增量更新本文件，不再新开重复规划文档。

## 当前进展（2026-03-22）

- 阶段 1 已完成：主计划文档落地、主文档集合收口、旧文档与旧脚本清理、实验路径改为仓库相对路径、基线提交已完成
- 已完成基线提交：`chore: snapshot xlstm-hybrid baseline and streamline repo`
- 阶段 2 已开始：新增多源 source registry，扩展 `hust / matr / oxford / pulsebat` adapter 与元数据字段
- 阶段 3 已持续推进：新增生命周期数据模块、`LifecycleHybridPredictor`、`LifecycleBiLSTMPredictor`、多源 `csv_paths` 训练支持、checkpoint warm-start 与 transfer benchmark 脚本
- 阶段 4 已推进：GraphRAG 已可消费 lifecycle evidence / model evidence，并补了一轮排序回归修复，减少不符合上下文的机理候选误排
- 阶段 5 已推进：后端已提供 `/api/v2/predict/lifecycle` 与 `/api/v2/explain/mechanism`，推理端支持 `final_release.json` 优先选权重

## 1. 外部趋势与执行依据

### 1.1 2023+ 文献趋势

当前路线以 2023 年之后的寿命预测与退化解释研究为依据，重点参考：

- [Predicting Battery Lifetime Under Varying Usage Conditions from Early Aging Data (2023)](https://arxiv.org/abs/2307.08382)
- [Prognosis of Multivariate Battery State of Performance and Health via Transformers (2023)](https://arxiv.org/abs/2309.10014)
- [Physics-Informed Machine Learning for Battery Degradation Diagnostics (2024)](https://arxiv.org/abs/2404.04429)
- [DiffBatt (2024)](https://arxiv.org/abs/2410.23893)
- [PBT: Pretrained Battery Transformer (2025)](https://arxiv.org/abs/2512.16334)
- [A Critical Review of AI-Based Battery RUL Prediction (2025)](https://www.mdpi.com/2313-0105/11/10/376)

### 1.2 对 BHMS 的直接启示

文献趋势已经非常明确：

- 不能只做单点 `RUL` 回归，必须升级为生命周期轨迹预测
- 不能只做单数据集实验，必须形成多源 benchmark 与跨域迁移能力
- 不能只给黑盒数值输出，必须补足机理解释、风险窗口与证据链
- 不能把 `Transformer` 当成唯一主线，`xLSTM` 在长程衰退建模上仍有明确保留价值

因此，BHMS 的工程目标固定为：

1. 建立统一生命周期预测任务
2. 以 `xLSTM + Transformer` 实现多任务混合架构
3. 以 `GraphRAG` 解释未来退化机理与风险归因
4. 形成答辩和论文都可复用的多源 benchmark 资产

## 2. 数据扩容路线

### 2.1 数据扩容基线

当前已接入：

- [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [CALCE Battery Data](https://calce.umd.edu/battery-data)
- Kaggle 演示型 CSV 数据

后续数据扩容基线固定为：

- [HUST 77-cell dataset](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
- [MATR / Severson 124-cell study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6386928/)
- [Oxford Battery Degradation Dataset 1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-8c52-e6f3db732d80)
- [PulseBat](https://arxiv.org/abs/2507.06278)

### 2.2 数据层级

#### A 层：主生命周期训练集

- `NASA`
- `CALCE`
- `HUST`
- `MATR`

#### B 层：轨迹增强集

- `Oxford`

#### C 层：诊断解释增强集

- `PulseBat`

### 2.3 接入顺序

1. `HUST`
2. `MATR`
3. `Oxford`
4. `PulseBat`

### 2.4 数据接入约束

- 新 source 一律走新 adapter，不改坏旧的 `NASA/CALCE/Kaggle` adapter
- source 枚举扩展为：`nasa | calce | kaggle | hust | matr | oxford | pulsebat`
- 统一 schema 必补字段：
  - `chemistry`
  - `form_factor`
  - `protocol_id`
  - `charge_c_rate`
  - `discharge_c_rate`
  - `ambient_temp`
  - `nominal_capacity`
  - `eol_ratio`
  - `dataset_license`
- `data/processed` 仍保留 source 分目录，但为每个 source 增加：
  - `*_lifecycle_dataset_summary.json`
  - `*_lifecycle_target_config.json`
- 新增统一 source registry 和 dataset card，保证后续实验按来源可追踪

## 3. 生命周期任务定义

### 3.1 任务升级

当前任务从单点 `RUL` 回归升级为统一生命周期预测：

- 输入：前期观测循环序列
- 输出：
  - 未来 `capacity_ratio` 轨迹
  - `SOH` 轨迹
  - `RUL`
  - `EOL cycle`
  - `knee cycle`

### 3.2 生命周期样本构造

固定样本构造策略如下：

- 观测比例：`0.2 / 0.3 / 0.4`
- 默认主展示比例：`0.3`
- 未来轨迹重采样长度：`64`
- 主轨迹目标统一使用 `capacity_ratio`
- `SOH` 由 `capacity_ratio` 派生，不额外构造第二条监督序列
- `knee` 标签由 source-aware label builder 生成；无法稳定生成时采用 loss mask，不丢样本

## 4. 生命周期模型路线

### 4.1 固定模型名称

新主模型固定命名为 `LifecycleHybridPredictor`。

### 4.2 核心原则

- `xLSTM` 是主编码器，不允许在第一版里弱化为可有可无的支路
- `Transformer` 只负责全局依赖增强与跨阶段关系建模
- 生命周期 decoder 和多任务 heads 建立在融合后的表征之上

### 4.3 固定结构

#### 输入层

- 数值特征投影
- `source embedding`
- `chemistry embedding`
- `protocol embedding`

#### 编码器

- `xLSTM branch`
- `Transformer branch`
- gated fusion

#### 解码器

- `LifecycleQueryDecoder`
- 64 个 learned query
- 非自回归 cross-attention trajectory decoder

#### 输出头

- `trajectory_head`
- `rul_head`
- `eol_head`
- `knee_head`
- `uncertainty_head`

### 4.4 固定损失函数

`L = 1.0*L_traj + 0.5*L_rul + 0.4*L_eol + 0.25*L_knee + 0.1*L_mono + 0.1*L_smooth + 0.1*L_domain`

### 4.5 固定训练策略

- Stage A：`NASA + CALCE + HUST + MATR` 多源预训练
- Stage B：先 `CALCE` 精调，再 `NASA` 精调
- `Oxford` 只参与 trajectory 辅助训练，不参与主 benchmark 排名

当前仓库中与该策略对应的落地产物：

- `configs/multisource_pretrain_hybrid.yaml`
- `configs/multisource_pretrain_bilstm.yaml`
- `configs/transfer_calce_hybrid.yaml`
- `configs/transfer_calce_bilstm.yaml`
- `configs/transfer_nasa_hybrid.yaml`
- `configs/transfer_nasa_bilstm.yaml`
- `scripts/run_transfer_benchmark.py`
- `scripts/promote_lifecycle_release.py`

### 4.6 主对照与消融

#### 主对照

- `LifecycleBiLSTM`
- `LifecycleHybridPredictor`

#### 固定消融

- `no_xlstm`
- `no_transformer`
- `no_domain_embedding`
- `no_trajectory_head`
- `single_source_only`

## 5. GraphRAG 升级路线

### 5.1 证据输入升级

GraphRAG 从“当前诊断”升级为“未来退化机理解释”，统一接收三类证据：

1. 当前异常证据
2. 生命周期预测证据
3. 模型解释证据

### 5.2 生命周期预测证据字段

固定包括：

- `predicted_knee_cycle`
- `predicted_eol_cycle`
- `accelerated_degradation_window`
- `future_capacity_fade_pattern`
- `temperature_risk`
- `resistance_risk`
- `voltage_risk`

### 5.3 知识图谱扩展

新增节点与关系：

- `Mechanism`
- `Protocol`
- `Chemistry`
- `RiskWindow`
- `Action`

知识库字段固定增加：

- `mechanisms`
- `stage_scope`
- `precursor_signals`
- `future_risk_patterns`
- `operating_condition_tags`
- `references`

### 5.4 候选机理排序规则

- `0.30 symptom_match`
- `0.25 future_risk_match`
- `0.20 stage_consistency`
- `0.15 source_scope`
- `0.10 threshold_hint_match`

### 5.5 LLM 使用策略

- 本地规则与图谱负责候选排序和证据链
- 外部 LLM 只负责 `llm_summary`
- 无 LLM 时必须完整降级到本地模板解释
- 不允许 LLM 直接参与候选机理打分

### 5.6 异常检测增强

- 规则链继续作为主证据来源
- `IsolationForest` 作为次级信号源，用于补充未知异常与置信增强
- 不允许完全替换统计规则检测

## 6. API、前端与报告迁移路线

### 6.1 API 固定顺序

先完成模型和 GraphRAG，再迁移接口。

#### 新主接口

- `POST /api/v2/predict/lifecycle`
- `POST /api/v2/explain/mechanism`

#### 兼容策略

- 保留 `/api/v1/predict/rul` 一轮迭代
- `v1` 内部使用 `v2` 生命周期结果派生 `RUL/EOL/projection`

### 6.2 前端迁移固定方向

- `Prediction` 页改成“全生命周期预测”
- `Diagnosis` 页改成“机理解释与未来风险归因”
- `Analysis` 页合并展示：
  - 生命周期轨迹
  - `knee/EOL`
  - 风险窗口
  - GraphRAG 证据链
  - LLM 摘要（可选）

### 6.3 报告导出升级

统一改成 lifecycle-first 报告，必须包含：

- 生命周期轨迹
- `knee/EOL/RUL`
- 风险窗口
- 候选机理
- 证据链
- 建议

## 7. 分阶段执行与提交计划

### 阶段 1：执行前整理、写入文档、强力清理、提交基线

#### 固定任务

- 写入主计划文档：`Doc/BHMS统一框架实施计划.md`
- `README.md` 增加主计划入口
- `Doc/BHMS毕业设计成品落地方案.md` 改成“当前仓库状态与封版优先级”定位
- 强力整理仓库结构，清理过时文档、脚本、缓存与误生成目录
- 实验 summary / metadata / checkpoint 引用改成仓库相对路径

#### 清理范围

保留主文档集合：

- `README.md`
- `Doc/BHMS统一框架实施计划.md`
- `Doc/系统使用说明.md`
- `Doc/实验复现说明.md`
- `Doc/答辩演示手册.md`
- `Doc/BHMS毕业设计成品落地方案.md`

删除或归档：

- 删除：`.DS_Store`、`__pycache__`、`.pytest_cache`、`frontend/dist`、`frontend/{src`
- 删除旧脚本：`scripts/train_mvp.py`、`scripts/prepare_nasa_data.py`、`scripts/fix_kaggle_setup.sh`
- 删除旧参考文档：
  - `Doc/BHMS_MVP实现说明.md`
  - `Doc/BHMS文档分析报告.md`
  - `Doc/BHMS系统开发文档_整合版.md`
  - `Doc/BHMS系统设计文档_整合版.md`
- 归档学校过程材料到：`Doc/archive/academic/`
- 保留优化资产：`configs/*_optimized.yaml`、`data/models/*/hybrid/optimized-config/`
- 删除探索性残留：`data/models/*/hybrid/search/`、非 canonical 的 `optimized/` 目录

#### 基线提交信息

`chore: snapshot xlstm-hybrid baseline and streamline repo`

### 阶段 2：数据扩容优先，建立真正可用的多源 benchmark

计划提交切分：

- `feat: add hust and matr dataset adapters`
- `feat: add oxford trajectory dataset support`
- `feat: add pulsebat diagnostic dataset support`

### 阶段 3：以 `xLSTM` 为中心的全生命周期模型重构

计划提交切分：

- `feat: add lifecycle data module and labels`
- `feat: add xlstm-centered lifecycle hybrid predictor`
- `feat: add multisource pretraining and ablation pipeline`

### 阶段 4：GraphRAG 从“当前诊断”升级到“未来退化机理解释”

计划提交切分：

- `feat: extend graphrag with lifecycle mechanism reasoning`
- `feat: add optional llm summary adapter for graphrag`

### 阶段 5：API、前端和报告迁移

计划提交切分：

- `feat: add lifecycle v2 api`
- `feat: migrate frontend to lifecycle workflow`
- `feat: upgrade case bundle and unified reports`

### 阶段 6：benchmark 固化与封版验收

计划提交切分：

- `chore: refresh lifecycle benchmarks and docs`
- 如需封版再补：`release: finalize bhms lifecycle baseline`

## 8. Benchmark 与验收主线

### 8.1 第一轮 benchmark 输出

#### within-source

- `CALCE`
- `NASA`
- `HUST`
- `MATR`

#### transfer

- `multisource pretrain -> CALCE fine-tune`
- `multisource pretrain -> NASA fine-tune`

### 8.2 展示主线

- `CALCE`：主展示生命周期预测与解释闭环
- `NASA`：主展示泛化与鲁棒性验证

### 8.3 阶段验收标准

- `CALCE` 生命周期预测、`knee/EOL/RUL`、GraphRAG 解释全通
- `NASA` 同口径全通
- `HUST/MATR` 至少导入成功并训练可跑通
- `LifecycleHybridPredictor` 在 `CALCE` 至少一项主指标优于 `LifecycleBiLSTM`
- GraphRAG 能消费 lifecycle 结果，不再只消费 anomaly event
- 仓库结构清晰、主文档收口、无明显历史噪音和重复入口

## 9. 测试与收口清单

### 仓库整理

- 删除项已无引用
- `README.md` 和 `Doc` 索引不再指向已删除文档
- `.gitignore` 能阻止缓存与残留目录回流

### 数据层

- `HUST / MATR / Oxford / PulseBat` adapter 可导入
- 多源 metadata 完整
- 生命周期标签跨 source 一致可用

### 模型层

- `LifecycleHybridPredictor` 输出 shape 固定正确
- 轨迹单调约束生效
- `no_xlstm` 消融能真实验证 `xLSTM` 主干价值

### 训练层

- 指标改为全测试集统一计算
- multi-seed summary / ablation summary 全部使用相对路径
- 多源预训练与单源精调可重复运行

### GraphRAG

- 无 LLM 时完整可用
- 有 LLM 时仅新增摘要，不改变证据链与排序
- 能解释未来退化阶段与风险窗口

### API / 前端

- `v2` 生命周期接口稳定
- `v1` 兼容层稳定
- 页面展示完整生命周期与机理解释链
- case bundle 可直接用于答辩与留档

## 10. 已锁定决策

- 当前已有未提交改动：全部纳入本轮工作基线
- 清理范围：强力整理
- 提交策略：多阶段细分提交
- 技术主线：坚持 `xLSTM` 为中心，不转纯 Transformer
- 数据优先级：先扩容，再做统一框架
- 文档记忆载体：只维护 `Doc/BHMS统一框架实施计划.md`
