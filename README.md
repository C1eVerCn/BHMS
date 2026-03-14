# BHMS - 锂电池健康管理系统

面向毕业设计答辩的 BHMS MVP：围绕 `NASA / CALCE / Kaggle` 三类数据源，打通“分来源导入 -> 周期级建模 -> Bi-LSTM 与 xLSTM-Transformer 对比训练 -> RUL 预测 -> 异常检测 -> GraphRAG 诊断 -> Web 展示”闭环。

## 当前实现范围

- 多数据源链路：支持 `NASA MAT`、`CALCE CSV`、`Kaggle CSV` 的来源级导入，统一映射为周期级 Schema 并落库到 SQLite
- 演示导入能力：前端可上传新的 `CSV / MAT` 文件，选择来源或自动识别，并决定是否标记为后续训练数据池候选
- 训练链路：按来源独立训练 `Bi-LSTM` 基线与 `xLSTM-Transformer` 混合模型，输出来源级 checkpoint、实验摘要与对比结果
- 推理链路：后端按电池来源优先加载该来源的 `Hybrid best checkpoint`，缺失时回退到该来源 `Bi-LSTM`，最后才使用启发式 fallback
- 诊断链路：异常检测输出标准化症状事件，诊断引擎从 `data/knowledge/battery_fault_knowledge.json` 读取本地知识库并生成报告

## 与整合文档的关系

`Doc` 目录中的设计/开发文档描述的是目标态全量系统，本次仓库实现的是“毕业设计 MVP”。
重点约束如下：

- 本轮不实现 JWT、Redis、Celery、监控、Docker 编排等工业化配套
- GraphRAG 采用本地结构化知识库 + 模板化文本生成，保留后续切换 Neo4j / LLM API 的接口边界
- xLSTM 的实现目标是验证混合架构可运行，不宣称线性计算复杂度
- 三个来源首轮按“分数据源独立训练”落地，不做混训

更多说明见 `Doc/BHMS_MVP实现说明.md`。

## 目录结构

```text
BHMS/
├── backend/                    # FastAPI backend
├── frontend/                   # React + Ant Design frontend
├── ml/                         # 数据处理、模型、训练与推理
├── kg/                         # GraphRAG 诊断引擎
├── configs/                    # 按来源/模型划分的训练配置
├── data/
│   ├── raw/
│   │   ├── nasa/               # NASA 原始 MAT 数据
│   │   ├── calce/              # CALCE 演示 CSV 数据
│   │   └── kaggle/             # Kaggle 演示 CSV 数据
│   ├── processed/              # 按来源隔离的预处理输出与元数据
│   ├── knowledge/              # 本地知识库
│   └── models/                 # 按来源隔离的模型权重、对比摘要
├── scripts/
└── tests/
```

## 环境要求

- Python 3.11+（本地验证使用 `Python 3.11`）
- Node.js 18+
- npm 9+

## 安装依赖

```bash
# Python
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Frontend
cd frontend
npm install
```

## 启动方式

### 1. 启动后端

```bash
source .venv/bin/activate
python backend/main.py
```

默认地址：`http://localhost:8000`

首次启动时，如果数据库为空，后端会自动尝试从 `data/raw/nasa`、`data/raw/calce`、`data/raw/kaggle` 导入可用样例数据。

### 2. 启动前端

```bash
cd frontend
npm run dev
```

默认地址：`http://localhost:3000`

## 数据准备与训练

### 按来源准备训练数据

```bash
source .venv/bin/activate
python scripts/prepare_datasets.py --source nasa --include-in-training
python scripts/prepare_datasets.py --source calce
python scripts/prepare_datasets.py --source kaggle
```

会在 `data/processed/<source>/` 下生成：

- `<source>_cycle_summary.csv`
- `<source>_split.json`
- `<source>_normalization.json`
- `<source>_feature_config.json`
- `<source>_dataset_summary.json`

### 按来源训练 Bi-LSTM / xLSTM-Transformer

```bash
source .venv/bin/activate
python scripts/train_models.py --source nasa --model bilstm --config configs/nasa_bilstm.yaml
python scripts/train_models.py --source nasa --model hybrid --config configs/nasa_hybrid.yaml
```

CALCE / Kaggle 同理：

```bash
python scripts/train_models.py --source calce --model bilstm --config configs/calce_bilstm.yaml
python scripts/train_models.py --source calce --model hybrid --config configs/calce_hybrid.yaml
python scripts/train_models.py --source kaggle --model bilstm --config configs/kaggle_bilstm.yaml
python scripts/train_models.py --source kaggle --model hybrid --config configs/kaggle_hybrid.yaml
```

### 生成来源级对比实验摘要

```bash
source .venv/bin/activate
python scripts/run_comparison.py --source nasa
python scripts/run_comparison.py --source calce
python scripts/run_comparison.py --source kaggle
```

训练结果会输出到 `data/models/<source>/`，包含：

- `bilstm/bilstm_best.pt` 与 `hybrid/hybrid_best.pt`
- 对应 JSON 元数据与训练摘要
- `comparison_summary.json`

## 主要接口

- `GET /api/v1/dashboard/summary`
- `GET /api/v1/batteries?page=1&page_size=10`
- `GET /api/v1/battery/{battery_id}/cycles?limit=120`
- `GET /api/v1/battery/{battery_id}/history`
- `GET /api/v1/battery/{battery_id}/health`
- `POST /api/v1/data/import-source`
- `POST /api/v1/data/upload`
- `POST /api/v1/predict/rul`
- `POST /api/v1/detect/anomaly`
- `POST /api/v1/diagnose`

所有业务响应统一返回：

```json
{
  "success": true,
  "message": "ok",
  "data": {}
}
```

## 测试与校验

```bash
# Python 语法检查
source .venv/bin/activate
python -m compileall backend kg ml tests scripts

# Python 单元测试
pytest tests/test_data_pipeline.py tests/test_anomaly_detector.py tests/test_graphrag.py tests/test_inference_service.py tests/test_models.py

# Frontend 构建
cd frontend && npm run build
```

## 后续扩展方向

- 接入真实 CALCE / Kaggle 更多格式子集，而不只是一套稳定演示格式
- 继续调优各来源 `Hybrid` 模型，完善论文级对照实验与消融实验
- 将本地知识库切换为 Neo4j + 更完整的知识抽取流程
- 增加认证、限流、监控、CI/CD 与部署脚本
