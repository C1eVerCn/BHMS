# BHMS 封版检查清单

本清单用于对当前仓库中的封版收口结果做人工复核，口径固定为“按交付价值分组”，不按 `git status` 顺序罗列。

## 事实来源与标记说明

- 事实来源：`git status --short`、`data/models/**` 下已落盘的 summary / release 文件、`data/processed/**` 生命周期元数据、`README.md` 与 `Doc/` 当前文档。
- 验证口径：
  - `已验证` = 已存在可解析的正式结果文件，或已完成系统级校验。
  - 当前封版执行记录中，`./.venv/bin/pytest -q` 为 `60 passed`，`frontend` 构建通过，`python scripts/normalize_repo_metadata_paths.py --check` 为 `changed=0`，`python scripts/validate_release_assets.py` 通过，8 份 lifecycle release manifest 的 checkpoint 解析 smoke 通过。
- “是否属于需要纳入版本控制的正式封版资产”表示建议进入正式封版提交集合；标记为“否”的内容更适合作为归档材料、运行追溯材料或按发布策略筛选纳入。

## 一、代码/配置改动

| 变更类别 | 关键路径/目录 | 是否已验证 | 是否属于需要纳入版本控制的正式封版资产 | 说明 |
| --- | --- | --- | --- | --- |
| 生命周期 transfer 训练主线 | `ml/training/lifecycle_transfer_runner.py`、`scripts/run_transfer_benchmark.py`、`ml/training/lifecycle_experiment_runner.py`、`ml/training/lifecycle_trainer.py`、`scripts/train_models.py` | 是 | 是 | 已支撑 `CALCE / NASA` 的 multisource pretrain -> fine-tune 实验，并落盘 4 份 transfer summary。 |
| final release 晋升与推理解析 | `scripts/promote_lifecycle_release.py`、`ml/inference/predictor.py`、`ml/training/experiment_artifacts.py` | 是 | 是 | 已形成 8 份 `release/final_release.json`，推理继续通过 `model_name=hybrid|bilstm` 显式选模型。 |
| 元数据相对路径与封版校验 | `scripts/normalize_repo_metadata_paths.py`、`scripts/validate_release_assets.py`、`ml/data/dataset.py` | 是 | 是 | 已把 `data/models/**`、`data/processed/**` 的路径元数据纳入相对路径清理与脚本化校验流程。 |
| 生命周期数据/训练口径补强 | `ml/data/lifecycle.py`、`ml/training/__init__.py` | 是 | 是 | 对齐 within-source 与 transfer 的 summary、artifact 与 release 解析路径。 |
| 回归修复与测试补强 | `kg/graphrag_engine.py`、`tests/test_graphrag.py`、`tests/test_experiment_pipeline.py`、`tests/test_inference_service.py` | 是 | 是 | GraphRAG 回归点已修复并重新纳入测试，当前 Python 测试全绿。 |
| 封版文档与说明收口 | `README.md`、`Doc/BHMS封版检查清单.md`、`Doc/BHMS最终封版说明.md` 以及 `Doc/` 下现有使用/复现/答辩文档 | 是 | 是 | 统一“lifecycle-first 工程封版资产收口”的交付口径，供提交、答辩与交接直接引用。 |
| 封版配置新增 | `configs/multisource_pretrain_hybrid.yaml`、`configs/multisource_pretrain_bilstm.yaml`、`configs/transfer_calce_hybrid.yaml`、`configs/transfer_calce_bilstm.yaml`、`configs/transfer_nasa_hybrid.yaml`、`configs/transfer_nasa_bilstm.yaml` | 是 | 是 | 已被真实 transfer benchmark 使用，不再是文档计划项。 |

## 二、实验资产新增/更新

### 2.1 建议纳入版本控制的正式封版资产

| 变更类别 | 关键路径/目录 | 是否已验证 | 是否属于需要纳入版本控制的正式封版资产 | 说明 |
| --- | --- | --- | --- | --- |
| `CALCE / NASA` transfer 正式汇总 | `data/models/calce/{hybrid,bilstm}/transfer/multisource_to_calce/`、`data/models/nasa/{hybrid,bilstm}/transfer/multisource_to_nasa/` | 是 | 是 | 包含 4 份 `*_transfer_summary.json`、对应 `plots/`，其 `best_checkpoint.path` 已同步指向 `release/checkpoints/` 下的正式权重。 |
| `HUST / MATR` 双模型 multi-seed 汇总 | `data/models/hust/{hybrid,bilstm}/*_multi_seed_summary.json`、`data/models/matr/{hybrid,bilstm}/*_multi_seed_summary.json` | 是 | 是 | `HUST / MATR` 现已具备双模型 within-source 主汇总，可用于扩展验证展示。 |
| `HUST / MATR` ablation 与 comparison | `data/models/hust/ablation_summary.json`、`data/models/matr/ablation_summary.json`、`data/models/hust/comparison_summary.json`、`data/models/matr/comparison_summary.json`、`data/models/{hust,matr}/plots/` | 是 | 是 | 补齐了原先缺失的 ablation / comparison 面，便于论文图表和答辩对照展示。 |
| `data/processed` 生命周期元数据刷新 | `data/processed/<source>/` 下 dataset summary、feature config 与 lifecycle summary | 是 | 是 | 仓库默认基线元数据已经统一到仓库相对路径口径，应与模型 summary 一起保留，避免封版后口径漂移。 |
| final release manifest | `data/models/{calce,nasa,hust,matr}/{hybrid,bilstm}/release/final_release.json` | 是 | 是 | 是推理侧优先读取的正式 release 入口，也是本轮封版最关键的交付资产。 |

### 2.2 建议单独归档或按发布策略筛选的中间训练产物

| 变更类别 | 关键路径/目录 | 是否已验证 | 是否属于需要纳入版本控制的正式封版资产 | 说明 |
| --- | --- | --- | --- | --- |
| per-seed checkpoints 与 runs 目录 | `data/models/*/*/runs/`、`data/models/*/*/transfer/*/{pretrain,fine_tune}/seed-*` | 是 | 否 | 适合作为复核与追溯材料，应通过 `scripts/archive_experiment_artifacts.py` 统一转入 `data/archive/experiments/<date>/`。 |
| 单次训练 checkpoint 与训练快照 | `data/models/*/*/*_best.pt`、`data/models/*/*/*_final.pt`、`data/models/*/*/*_best.json`、`data/models/*/*/*_final.json` | 是 | 否 | 可辅助追溯最佳权重来源；若提交体积受限，可保留 manifest 与 summary，单独归档权重。 |
| 训练明细与测试明细 | `data/models/*/*/training_summary.json`、`data/models/*/*/test_details.json` | 是 | 否 | 建议保留在归档包中，用于答辩追问或实验核查，不作为正式封版最小面。 |

## 三、final release 产物

下表覆盖当前已纳入 final release 的全部 `source/model` 组合。校验标准为：`final_release.json` 存在、`summary_path` 指向真实 summary、`checkpoint_path` 能解析到真实 checkpoint。

| 变更类别 | 关键路径/目录 | 是否已验证 | 是否属于需要纳入版本控制的正式封版资产 | 说明 |
| --- | --- | --- | --- | --- |
| `calce / hybrid` final release | `data/models/calce/hybrid/release/final_release.json` | 是 | 是 | `release_label=transfer-final-2026-03-22`，来源于 `data/models/calce/hybrid/transfer/multisource_to_calce/hybrid_transfer_summary.json`，checkpoint 可解析。 |
| `calce / bilstm` final release | `data/models/calce/bilstm/release/final_release.json` | 是 | 是 | `release_label=transfer-final-2026-03-22`，来源于 `data/models/calce/bilstm/transfer/multisource_to_calce/bilstm_transfer_summary.json`，checkpoint 可解析。 |
| `nasa / hybrid` final release | `data/models/nasa/hybrid/release/final_release.json` | 是 | 是 | `release_label=transfer-final-2026-03-22`，来源于 `data/models/nasa/hybrid/transfer/multisource_to_nasa/hybrid_transfer_summary.json`，checkpoint 可解析。 |
| `nasa / bilstm` final release | `data/models/nasa/bilstm/release/final_release.json` | 是 | 是 | `release_label=transfer-final-2026-03-22`，来源于 `data/models/nasa/bilstm/transfer/multisource_to_nasa/bilstm_transfer_summary.json`，checkpoint 可解析。 |
| `hust / hybrid` final release | `data/models/hust/hybrid/release/final_release.json` | 是 | 是 | `release_label=within-source-final-2026-03-22`，来源于 `data/models/hust/hybrid/hybrid_multi_seed_summary.json`，checkpoint 可解析。 |
| `hust / bilstm` final release | `data/models/hust/bilstm/release/final_release.json` | 是 | 是 | `release_label=within-source-final-2026-03-22`，来源于 `data/models/hust/bilstm/bilstm_multi_seed_summary.json`，checkpoint 可解析。 |
| `matr / hybrid` final release | `data/models/matr/hybrid/release/final_release.json` | 是 | 是 | `release_label=within-source-final-2026-03-22`，来源于 `data/models/matr/hybrid/hybrid_multi_seed_summary.json`，checkpoint 可解析。 |
| `matr / bilstm` final release | `data/models/matr/bilstm/release/final_release.json` | 是 | 是 | `release_label=within-source-final-2026-03-22`，来源于 `data/models/matr/bilstm/bilstm_multi_seed_summary.json`，checkpoint 可解析。 |

## 四、封版复核结论

- `CALCE / NASA` 的 transfer summary 已全部落盘，`suite_kind=transfer` 的正式结果文件已具备。
- `HUST / MATR` 的 within-source 缺口已补齐，`bilstm` multi-seed、`hybrid` ablation、comparison 均已存在。
- `CALCE / NASA / HUST / MATR` 的 `hybrid` 与 `bilstm` 均已有独立 `final_release.json`，当前保持“双模型并存”口径，不引入新的全局默认模型配置。
- `data/models/**` 与 `data/processed/**` 当前已经完成相对路径收口，并可通过 `scripts/validate_release_assets.py` 重复校验。
- `Kaggle / Oxford / PulseBat` 本轮没有纳入 final release 集：`Kaggle` 保留为基线/对照资产，`Oxford` 仍为辅助 trajectory 源，`PulseBat` 仍为 enhancement-only 机制源。
