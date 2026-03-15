# BHMS 系统开发文档（参考版）

更新日期：2026-03-15

## 重要说明

这份文档现在的定位是“开发参考与目标态说明”，不是当前实现状态的唯一真值。

请按下面优先级理解仓库文档：

1. `README.md`：当前真实能力与运行入口
2. `Doc/系统使用说明.md`：启动与页面使用
3. `Doc/实验复现说明.md`：实验与案例资产复现
4. 本文：开发参考、扩展方向与历史设计背景

## 当前真实开发基线

当前仓库实际使用的技术栈为：

- 后端：FastAPI + SQLite
- 前端：React + Vite + Ant Design + ECharts
- 模型：PyTorch
- 图谱：Neo4j / memory 双后端
- 数据：`NASA / CALCE / Kaggle` 三源公开数据

当前已经完成的开发事项：

- 三源导入与周期级 schema 统一
- RUL 预测、异常检测、GraphRAG 诊断
- 多 seed 实验与消融实验
- 数据画像、知识库摘要、案例目录导出
- 成品流程测试、前端构建和后端单测

## 不应视为当前已交付内容的条目

以下能力可以保留为长期扩展方向，但不要当作当前仓库已完成项：

- Redis / Celery 任务队列
- 企业级 JWT / RBAC / 审计体系
- Prometheus / Grafana 监控
- CI/CD 与生产化容器编排
- 在线 LLM 服务治理与成本控制

## 当前开发建议

### 代码优先级

- 优先修正实验质量与复现性
- 其次清理 `data/models/**` 和案例资产中的绝对路径
- 最后才是扩新页面或工业化配套

### 数据优先级

- `scripts/refresh_processed_baselines.py` 负责仓库默认 `data/processed` 基线
- `scripts/prepare_datasets.py` 负责当前训练池导出
- 不要再把本地训练池状态直接当作仓库应提交基线

### 前端优先级

- 继续复用 `PageHero / InsightCard / PanelCard / StatusTag` 这组组件模式
- 避免在现有设计语言之外继续堆砌新风格页面

## 参考阅读

- `frontend/docs/ui-guidelines.md`
- `Doc/BHMS毕业设计成品落地方案.md`
- `Doc/答辩演示手册.md`

## 后续扩展建议

如果未来要继续沿本文扩展，建议按下面顺序：

1. 清理实验资产绝对路径
2. 优化模型效果并沉淀论文结论页
3. 封装最小一键复现链路
4. 再考虑任务队列、监控、工业部署等加分项
