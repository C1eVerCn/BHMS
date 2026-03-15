# BHMS 毕业设计成品落地方案

更新日期：2026-03-15

## 当前状态结论

截至 2026-03-15，BHMS 已经完成从“原型 / MVP”向“毕业设计成品基线”的跃迁：

> 系统已具备三源导入、RUL 预测、异常检测、GraphRAG 诊断、多 seed / ablation 实验、分析中心、Markdown 报告与案例目录导出闭环；真正仍待补强的是实验质量和路径可复现性，而不是页面或脚本是否存在。

## 已完成能力

### 成品闭环

- 多源导入：支持 `NASA MAT`、`CALCE CSV`、`Kaggle CSV`
- 推理链路：预测、异常检测、GraphRAG 诊断、报告导出已全通
- 前端展示：分析中心包含 `RUL 分析`、`GraphRAG 诊断`、`训练与实验`、`数据画像`、`案例导出`
- 案例闭环：支持只读预览和目录化导出，缺预测/诊断时自动补生成

### 实验系统

- `Bi-LSTM` 与 `Hybrid` 均已支持三 seed 汇总
- `no_xlstm`、`no_transformer`、`capacity_only` 消融已支持三 seed 聚合
- 来源级图表、plot metadata、实验概览、实验详情、ablation 接口已完成

### 领域解释能力

- 知识库已扩展到 12 类故障模式
- 异常检测支持来源自适应阈值与更细的异常等级
- 诊断结果包含 `rule_id`、`evidence_source`、`confidence_basis`、`source_scope`、`decision_basis`

## 当前真正短板

### 科研有效性

- 多个来源上的 `R²` 仍为负值
- 某些来源上 `Hybrid` 并未稳定优于 baseline
- 论文级结论目前仍需要围绕 split、特征组合和超参数继续补强

### 可复现性

- `data/models/**` 与部分案例资产里仍存在机器绝对路径
- 这会影响换机复现、仓库审查和论文材料移植

### 交付封版度

- 还缺一条完整的一键式复现链路，把 `Neo4j -> processed baseline -> full suite -> case export` 串成固定命令流
- 论文固定图表页和最终讲稿仍需继续沉淀

## 本轮收尾后仓库的标准状态

- `README / Doc / 系统页面` 三者口径一致
- `data/processed/<source>` 可由 `data/raw/<source>` 确定性重建
- `data/demo_uploads` 作为演示资产保留，但不会污染仓库默认 processed 基线
- 工作树保持干净，不再残留“应该提交但没提交”的交付材料

## 接下来最重要的三件事

### 1. 清理 `data/models/**` 的绝对路径

这是下一轮最优先的可复现性问题。需要把 multi-seed summary、ablation summary、plot metadata、training summary、run-level artifacts 统一改为仓库相对路径。

### 2. 提升实验质量

重点不是继续扩页面，而是让至少一个来源形成更可信的结论：

- 复核 split 是否合理
- 检查 `capacity_only` 与完整特征表现反常的原因
- 重新调 `seq_len / hidden size / dropout / learning rate / epoch / patience`
- 争取形成一个“Hybrid 相对 baseline 有稳定收益”的来源级结论

### 3. 封装一条最小复现命令链

建议下一轮固定为：

1. `docker compose -f docker-compose.neo4j.yml up -d`
2. `python scripts/init_neo4j_graph.py`
3. `python scripts/refresh_processed_baselines.py --sources nasa calce kaggle`
4. 运行 full suite
5. 导出一个标准案例目录

## 结论

BHMS 当前已经具备答辩成品骨架，短板不在“有没有做”，而在“做得是否足够有说服力”。后续开发优先级应固定为：

`实验质量 > 资产路径可复现 > 一键封版 > 新页面/新功能`
