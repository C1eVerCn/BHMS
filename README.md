# BHMS - 锂电池健康管理系统

面向毕业设计答辩与论文留档的 BHMS 成品基线版。当前仓库已经完成 `NASA / CALCE / Kaggle` 三源导入、RUL 预测、异常检测、GraphRAG 诊断、多 seed 实验、消融实验、分析中心展示、Markdown 报告与案例目录导出闭环。

## 当前定位

仓库的目标不是工业级部署，而是完成一套 `可答辩、可复现、可解释` 的毕业设计成品：

- `可答辩`：支持 8 分钟内稳定演示 `导入 -> 预测 -> 诊断 -> 分析 -> 导出`
- `可复现`：保留三源 `data/processed` 基线、训练脚本、多 seed / ablation 汇总与案例目录导出
- `可解释`：系统内可查看 RUL 证据链、候选故障排序、GraphRAG 子图、decision basis、知识库摘要
- `轻量工程化`：继续使用 `FastAPI + React/Vite + SQLite + PyTorch + Neo4j/内存图谱`

当前最主要的短板不是脚本缺失，而是实验指标仍偏弱：多个来源上的 `R²` 仍为负，论文级结论还需要继续补强。

## 当前实现范围

- 多源导入：支持 `NASA MAT`、`CALCE CSV`、`Kaggle CSV`
- 数据持久化：统一映射为周期级 schema 并落库到 SQLite
- 预测链路：返回 RUL、寿命投影、关键特征贡献、关键时间窗口贡献、置信度说明和 Markdown 报告
- 诊断链路：返回异常事件、候选故障排序、GraphRAG 子图、根因链、建议、decision basis 和 Markdown 报告
- 训练链路：支持 `Bi-LSTM`、`Hybrid`、三 seed 汇总、ablation 汇总、来源级图表与实验概览接口
- 分析中心：包含 `RUL 分析`、`GraphRAG 诊断`、`训练与实验`、`数据画像`、`案例导出`
- 案例闭环：支持 `GET /api/v1/reports/case-bundle/{battery_id}` 预览与 `POST /api/v1/reports/case-bundle/{battery_id}/export` 目录导出

## 项目结构

```text
BHMS/
├── backend/                    # FastAPI backend
├── frontend/                   # React + Vite + Ant Design frontend
├── ml/                         # 数据处理、模型、训练与推理
├── kg/                         # GraphRAG 诊断引擎
├── scripts/                    # 数据准备、训练、图谱初始化、基线刷新脚本
├── tests/                      # 单测与成品流程测试
├── data/
│   ├── raw/                    # 原始公开数据
│   ├── processed/              # 仓库默认训练基线，可由 data/raw 重建
│   ├── knowledge/              # 故障知识库 JSON
│   ├── models/                 # 实验资产、图表、checkpoint
│   └── exports/                # 案例目录导出
└── Doc/                        # 使用、复现、答辩与参考文档
```

## 快速开始

### 1. Python 与前端依赖

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

cd frontend
npm install
```

### 2. 启动 Neo4j（推荐）

```bash
docker compose -f docker-compose.neo4j.yml up -d
source .venv/bin/activate
python scripts/init_neo4j_graph.py
```

如果现场不方便启动 Neo4j，可把 `BHMS_GRAPH_BACKEND` 设为 `memory` 继续演示。

### 3. 刷新仓库默认 `data/processed` 基线

```bash
source .venv/bin/activate
python scripts/refresh_processed_baselines.py --sources nasa calce kaggle
```

这一步只从 `data/raw/<source>` 重建仓库默认基线，不读取 SQLite 当前训练池，也不会把 `data/demo_uploads` 的未见样本写回 `data/processed`。

### 4. 启动后端与前端

```bash
source .venv/bin/activate
python backend/main.py
```

```bash
cd frontend
npm run dev
```

默认前端地址：`http://localhost:3000`

## 标准演示流

建议固定使用下面这条链路：

1. 上传一个未见样本，优先 `data/demo_uploads/calce/calce_unseen_fault_demo.csv`
2. 点击“立即预测与诊断”
3. 在分析中心查看：
   - `RUL 分析`：轨迹、EOL、贡献解释、预测报告
   - `GraphRAG 诊断`：候选故障、子图、decision basis、诊断报告
   - `训练与实验`：多 seed、ablation、plot 清单、来源级 headline
   - `数据画像`：来源统计、split、特征范围、知识库摘要、演示预设
   - `案例导出`：案例完整性、最近导出目录、文件清单
4. 导出预测报告、诊断报告、案例目录

## 数据与实验命令

### 仓库默认基线

```bash
python scripts/refresh_processed_baselines.py --sources nasa calce kaggle
```

### 本地训练池导出

```bash
python scripts/prepare_datasets.py --source calce --include-in-training
```

`prepare_datasets.py` 用于从当前 SQLite 训练池导出本地训练数据；`refresh_processed_baselines.py` 才是仓库默认基线生成入口。

### 多 seed 与消融

```bash
python scripts/run_multi_seed_experiment.py --source nasa --model bilstm --config configs/nasa_bilstm.yaml --force
python scripts/run_multi_seed_experiment.py --source nasa --model hybrid --config configs/nasa_hybrid.yaml --force
python scripts/run_ablation_study.py --source nasa --config configs/nasa_hybrid.yaml --force
python scripts/run_comparison.py --source nasa
```

同样的命令可替换 `source=calce` 或 `source=kaggle`。

## 关键产物

- 仓库默认基线：`data/processed/<source>/`
- 多 seed 汇总：`data/models/<source>/<model>/<model>_multi_seed_summary.json`
- 消融汇总：`data/models/<source>/ablation_summary.json`
- 来源级图表：`data/models/<source>/plots/`
- 案例目录：`data/exports/cases/<battery_id>/<timestamp>/`

## 已知限制

- 多个来源上的 `R²` 仍偏弱，论文级实验结论需要继续强化
- `data/models/**` 与部分实验元数据仍包含机器绝对路径，这是下一轮最高优先级的可复现性问题
- 当前目标不是工业级认证、监控、队列或集群部署

## 文档索引

- `Doc/系统使用说明.md`：新机器启动与页面使用
- `Doc/实验复现说明.md`：数据、训练、实验与案例资产复现
- `Doc/答辩演示手册.md`：5-8 分钟演示流程与答辩口径
- `Doc/BHMS毕业设计成品落地方案.md`：当前状态与下一步优先级
- `Doc/BHMS_MVP实现说明.md`：MVP 与当前成品基线的边界说明
