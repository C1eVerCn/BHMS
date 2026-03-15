# BHMS 系统设计文档（参考版）

更新日期：2026-03-15

## 重要说明

这份文档现在定位为“设计参考与目标态说明”，不是当前实现状态的唯一依据。

当前真实系统状态请优先以 `README.md`、`Doc/系统使用说明.md` 和 `Doc/实验复现说明.md` 为准。

## 当前实际系统架构

### 表现层

- React + Vite 前端
- 首页 / 电池档案 / 数据导入 / RUL 预测 / 故障诊断 / 分析中心 六页结构

### 服务层

- FastAPI 提供 `/api/v1` 接口
- 覆盖电池、上传、预测、异常、诊断、训练、报告、洞察等路由

### 算法层

- `Bi-LSTM` 基线
- `Hybrid`（xLSTM + Transformer）
- 异常检测
- GraphRAG 诊断引擎

### 数据层

- 原始数据：`data/raw/<source>`
- 仓库默认训练基线：`data/processed/<source>`
- 实验资产：`data/models/<source>`
- 案例资产：`data/exports/cases/<battery_id>/<timestamp>`
- 持久化：SQLite

## 当前设计重点

### 1. 解释性优先

系统不仅输出预测和诊断结果，还输出：

- 关键特征贡献
- 关键时间窗口贡献
- 候选故障排序
- 图谱子图
- `rule_id / source_scope / confidence_basis / decision_basis`

### 2. 复现性优先

- `data/processed` 通过 `scripts/refresh_processed_baselines.py` 从 `data/raw` 重建
- 多 seed / ablation 产物以 JSON + PNG + metadata 落盘
- 案例导出目录可直接作为论文附录素材

### 3. 成品演示优先

系统当前服务的主要场景是：

- 毕业设计答辩
- 论文截图与附录
- 本地可复现实验展示

而不是工业在线服务。

## 仍属目标态的设计项

下面这些仍属于远期设计，不应回推成当前仓库“已完成”：

- 多租户鉴权与审计
- 高并发队列与异步任务基础设施
- 企业级监控、告警与运维
- 集群化 Neo4j 与在线 LLM 编排

## 设计上的下一步

建议下一轮优先解决：

1. `data/models/**` 的绝对路径清理
2. 实验效果与论文结论增强
3. 一键复现命令链封装

完成这三项后，再考虑更大的架构升级。
