# BHMS - 锂电池健康管理系统

面向毕业设计答辩与论文留档的 BHMS 生命周期基线版。当前仓库已经完成 `lifecycle-first` 的系统主干：`NASA / CALCE / Kaggle` 基线导入、`HUST / MATR / Oxford / PulseBat` 多源适配、生命周期预测、GraphRAG 机理解释、多 seed / ablation / transfer 实验、Markdown 报告与案例目录导出闭环。

## 当前定位

仓库的目标不是工业级部署，而是完成一套 `可答辩、可复现、可解释` 的毕业设计成品：

- `可答辩`：支持 8 分钟内稳定演示 `导入 -> 预测 -> 诊断 -> 分析 -> 导出`
- `可复现`：保留生命周期 `data/processed` 基线、训练脚本、多 seed / ablation / transfer 汇总与案例目录导出
- `可解释`：系统内可查看 lifecycle 证据链、候选故障排序、GraphRAG 子图、decision basis、知识库摘要
- `轻量工程化`：继续使用 `FastAPI + React/Vite + SQLite + PyTorch + Neo4j/内存图谱`

当前仓库已经完成 `lifecycle-first` 工程封版资产收口，可作为毕业设计提交、答辩展示和后续复核的统一基线。需要注意的是：当前结论仍然是“研究原型/工程封版”口径，而不是工业级部署口径；其中 `NASA` transfer 已真实跑通但指标仍弱，不应夸大为强泛化结论。

## 论文级状态

- 当前论文门槛尚未通过，`Hybrid 全面优于 BiLSTM` 还不能写入论文主结论。
- 当前 7 个核心 benchmark 单元中，未通过的是：
  - `CALCE / within-source`：Hybrid `RMSE=0.021945`、`R²=0.513922`；BiLSTM `RMSE=0.017631`、`R²=0.695668`
  - `Kaggle / within-source`：Hybrid `RMSE=0.027337`、`R²=0.322002`；BiLSTM `RMSE=0.024020`、`R²=0.512857`
- 当前消融门槛未通过的来源：`NASA / CALCE / MATR`
- 统一论文真值来源固定为：
  - `data/models/<source>/<model>/*_multi_seed_summary.json`
  - `data/models/<source>/<model>/transfer/multisource_to_<source>/<model>_transfer_summary.json`
  - `Doc/BHMS论文证据包.md`
  - `Doc/BHMS论文证据包.json`
- 在 `python scripts/validate_release_assets.py --require-paper-gate` 返回通过前，论文正文应保持“继续优化中”表述。

## 当前实现范围

- 多源导入：支持 `NASA / CALCE / Kaggle / HUST / MATR / Oxford / PulseBat`
- 数据层分级：
  - `NASA / CALCE / Kaggle / HUST / MATR`：生命周期训练集或 benchmark 集
  - `Oxford`：trajectory 辅助源
  - `PulseBat`：enhancement-only 机制增强源，不进入生命周期主训练
- 生命周期推理：返回 `trajectory / RUL / knee / EOL / uncertainty`、关键特征贡献、关键时间窗口贡献和 Markdown 报告
- 诊断链路：`/api/v2/predict/lifecycle` 与 `/api/v2/explain/mechanism` 已打通，GraphRAG 能消费 anomaly + lifecycle + model evidence
- 训练链路：支持 `LifecycleBiLSTM`、`LifecycleHybridPredictor`、多 seed、ablation、多源 `pretrain -> fine_tune` transfer benchmark 和 final release manifest 晋升
- 分析中心：包含 `Lifecycle 分析`、`GraphRAG 诊断`、`训练与实验`、`数据画像`、`案例导出`
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
python scripts/refresh_processed_baselines.py --sources nasa calce kaggle hust matr oxford pulsebat
```

这一步只从 `data/raw/<source>` 重建仓库默认基线，不读取 SQLite 当前训练池，也不会把 `data/demo_uploads` 的未见样本写回 `data/processed`。

说明：

- `Oxford` 会生成辅助 trajectory 资产，但不进入主 benchmark 排名
- `PulseBat` 会生成 enhancement asset manifest，不进入生命周期主训练

### 4. 启动后端与前端

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
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
python scripts/refresh_processed_baselines.py --sources nasa calce kaggle hust matr oxford pulsebat
```

### 本地训练池导出

```bash
python scripts/prepare_datasets.py --source calce --include-in-training
```

`prepare_datasets.py` 用于从当前 SQLite 训练池导出本地训练数据；`refresh_processed_baselines.py` 才是仓库默认基线生成入口。

### Within-source 多 seed 与消融

```bash
python scripts/run_multi_seed_experiment.py --source nasa --model bilstm --config configs/nasa_bilstm.yaml --force
python scripts/run_multi_seed_experiment.py --source nasa --model hybrid --config configs/nasa_hybrid.yaml --force
python scripts/run_ablation_study.py --source nasa --config configs/nasa_hybrid.yaml --force
python scripts/run_comparison.py --source nasa
```

同样的命令可替换为 `source=calce / kaggle / hust / matr`。其中 `HUST / MATR` 更适合作为扩展验证源。

### 多源 pretrain -> fine_tune transfer

```bash
python scripts/run_transfer_benchmark.py --target calce --model hybrid --pretrain-config configs/multisource_pretrain_hybrid.yaml --finetune-config configs/transfer_calce_hybrid.yaml --seeds 7,21,42
python scripts/run_transfer_benchmark.py --target nasa --model hybrid --pretrain-config configs/multisource_pretrain_hybrid.yaml --finetune-config configs/transfer_nasa_hybrid.yaml --seeds 7,21,42
python scripts/run_transfer_benchmark.py --target calce --model bilstm --pretrain-config configs/multisource_pretrain_bilstm.yaml --finetune-config configs/transfer_calce_bilstm.yaml --seeds 7,21,42
```

### 晋升最终 lifecycle release

```bash
python scripts/promote_lifecycle_release.py --source calce --model hybrid
python scripts/promote_lifecycle_release.py --source nasa --model hybrid --summary data/models/nasa/hybrid/transfer/multisource_to_nasa/hybrid_transfer_summary.json
```

推理服务会优先读取 `data/models/<source>/<model>/release/final_release.json`；release manifest 会固定指向 `release/checkpoints/` 下的正式权重。如果 release manifest 不存在，才会回退到 transfer / multi-seed / 单次实验 checkpoint。

### 封版收口与验收

```bash
python scripts/normalize_repo_metadata_paths.py
python scripts/rebuild_benchmark_truth.py
python scripts/validate_release_assets.py
python scripts/archive_experiment_artifacts.py --archive-label 2026-03-23 --dry-run
```

建议把下面三项作为固定封版验收矩阵：

- `./.venv/bin/pytest -q`
- `cd frontend && npm run build -- --outDir /tmp/bhms-frontend-build`
- `python scripts/validate_release_assets.py`
- `python scripts/validate_release_assets.py --require-paper-gate`（仅在准备把 “Hybrid 全面优于” 写入论文主结论时执行）

## 关键产物

- 仓库默认基线：`data/processed/<source>/`
- 多 seed 汇总：`data/models/<source>/<model>/<model>_multi_seed_summary.json`
- transfer 汇总：`data/models/<source>/<model>/transfer/multisource_to_<source>/<model>_transfer_summary.json`
- final release manifest：`data/models/<source>/<model>/release/final_release.json`
- formal release checkpoint：`data/models/<source>/<model>/release/checkpoints/<model>_release.pt`
- 消融汇总：`data/models/<source>/ablation_summary.json`
- 来源级图表：`data/models/<source>/plots/`
- benchmark 真值重建：`data/models/<source>/comparison_summary.json`
- 论文证据包：`Doc/BHMS论文证据包.md`、`Doc/BHMS论文证据包.json`
- 运行时案例目录：`data/exports/cases/<battery_id>/<timestamp>/`（可清空后按需重新导出）
- 中间实验归档：`data/archive/experiments/<date>/`

## 已知限制

- 多个来源上的 `R²` 仍偏弱，论文级实验结论需要继续强化
- transfer 配置、脚本与 final release manifest 已经落地，但论文级 benchmark 结论仍需要持续强化与人工审核
- 中间训练材料默认转入 `data/archive/experiments/<date>/`，不再作为正式封版资产留在主工作区
- 当前目标不是工业级认证、监控、队列或集群部署

## 文档索引

- `Doc/系统使用说明.md`：新机器启动与页面使用
- `Doc/BHMS前后端接口梳理.md`：前后端页面、状态层、接口与模型调用入口总览
- `Doc/实验复现说明.md`：数据、训练、实验与案例资产复现
- `Doc/答辩演示手册.md`：5-8 分钟演示流程与答辩口径
- `Doc/BHMS论文证据包.md`：当前核心 benchmark 矩阵、论文门槛与消融门槛汇总
- `Doc/BHMS统一框架实施计划.md`：统一框架主线、阶段目标与后续实施路线
- `Doc/BHMS封版检查清单.md`：封版资产、验证状态与正式提交范围核对清单
- `Doc/BHMS最终封版说明.md`：本轮封版结论、benchmark 结果与 final release 覆盖范围
