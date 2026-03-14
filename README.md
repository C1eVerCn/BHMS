# BHMS - 锂电池健康管理系统

面向毕业设计答辩的 BHMS MVP：基于真实 NASA 电池老化数据，打通“数据导入 -> 周期级建模 -> RUL 预测 -> 异常检测 -> GraphRAG 诊断 -> Web 展示”闭环。

## 当前实现范围

- 真实数据链路：仓库内 `data/raw/nasa` 的 MAT 原始数据可直接预处理为周期级 CSV 并自动导入 SQLite
- 模型链路：支持 `Bi-LSTM` 基线与 `xLSTM-Transformer` 混合模型训练；推理支持训练权重与启发式 fallback
- 诊断链路：异常检测输出标准化症状事件，诊断引擎从 `data/knowledge/battery_fault_knowledge.json` 读取本地知识库并生成报告
- 后端：FastAPI + SQLite，提供电池列表、周期数据、健康状态、预测、异常、诊断、NASA 导入、CSV 上传等接口
- 前端：React + Ant Design，已替换掉主要 Mock 页面，直接消费真实 API

## 与整合文档的关系

`Doc` 目录中的设计/开发文档描述的是目标态全量系统，本次仓库实现的是“毕业设计 MVP”。
重点约束如下：

- 本轮不实现 JWT、Redis、Celery、监控、Docker 编排等工业化配套
- GraphRAG 采用本地结构化知识库 + 模板化文本生成，保留后续切换 Neo4j / LLM API 的接口边界
- xLSTM 的实现目标是验证混合架构可运行，不宣称线性计算复杂度

更多说明见 `Doc/BHMS_MVP实现说明.md`。

## 目录结构

```text
BHMS/
├── backend/                    # FastAPI MVP backend
│   └── app/
├── frontend/                   # React + Ant Design frontend
├── ml/                         # 数据处理、模型、训练与推理
├── kg/                         # GraphRAG 诊断引擎
├── data/
│   ├── raw/nasa/               # NASA 原始 MAT 数据
│   ├── processed/              # 预处理输出
│   ├── knowledge/              # 本地知识库
│   └── models/                 # 模型权重与训练摘要
├── scripts/
└── tests/
```

## 环境要求

- Python 3.10+
- Node.js 18+
- npm 9+

## 安装依赖

```bash
# Python
python3 -m venv .venv
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

首次启动时，如果数据库为空，后端会自动尝试从 `data/raw/nasa` 解析并导入示例数据。

### 2. 启动前端

```bash
cd frontend
npm run dev
```

默认地址：`http://localhost:3000`

## 数据准备与训练

### 预处理 NASA 数据

```bash
source .venv/bin/activate
python scripts/prepare_nasa_data.py --output data/processed/nasa_cycle_summary.csv
```

### 训练 Bi-LSTM 基线

```bash
source .venv/bin/activate
python scripts/train_mvp.py --model bilstm --data data/processed/nasa_cycle_summary.csv
```

### 训练 xLSTM-Transformer

```bash
source .venv/bin/activate
python scripts/train_mvp.py --model hybrid --data data/processed/nasa_cycle_summary.csv
```

训练结果会输出到 `data/models`，包含：

- `bilstm_best.pt` / `hybrid_best.pt`
- 对应 JSON 摘要
- TensorBoard 日志

## 主要接口

- `GET /api/v1/dashboard/summary`
- `GET /api/v1/batteries?page=1&page_size=10`
- `GET /api/v1/battery/{battery_id}/cycles?limit=120`
- `GET /api/v1/battery/{battery_id}/history`
- `GET /api/v1/battery/{battery_id}/health`
- `POST /api/v1/data/import-nasa`
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
python3 -m compileall backend kg ml tests scripts

# Frontend 类型检查
cd frontend && ./node_modules/.bin/tsc --noEmit
```

如已安装完整 Python 依赖，可继续运行：

```bash
pytest -q
```

## 后续扩展方向

- 用真实训练权重替换启发式 fallback 推理
- 将本地知识库切换为 Neo4j + 更完整的知识抽取流程
- 增加认证、限流、监控、CI/CD 与部署脚本
- 增加更细粒度的 Battery Detail 页面与案例回放页面
