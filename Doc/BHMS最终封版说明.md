# BHMS 最终封版说明

## 一句话结论

BHMS 当前已经完成 `lifecycle-first` 的系统封版与论文证据面重建：核心 benchmark、transfer、ablation、comparison、release manifest 与论文证据包都已可复核；但截至 `2026-04-03`，论文门槛仍未通过，因此系统可以封版，论文主结论还不能写成“Hybrid 全面优于 BiLSTM”。

## 本轮新增完成项

- 完成 `CALCE / NASA` 四组真实 transfer benchmark，并生成对应 `hybrid` / `bilstm` transfer summary 与 plots：
  - `data/models/calce/hybrid/transfer/multisource_to_calce/hybrid_transfer_summary.json`
  - `data/models/calce/bilstm/transfer/multisource_to_calce/bilstm_transfer_summary.json`
  - `data/models/nasa/hybrid/transfer/multisource_to_nasa/hybrid_transfer_summary.json`
  - `data/models/nasa/bilstm/transfer/multisource_to_nasa/bilstm_transfer_summary.json`
- 补齐 `HUST / MATR` 的 within-source 缺口：
  - `bilstm` multi-seed summary 已补齐；
  - `hybrid` ablation 已补齐；
  - comparison summary 已刷新。
- 晋升 8 份 lifecycle final release manifest，覆盖 `CALCE / NASA / HUST / MATR` x `hybrid / bilstm`：
  - transfer 来源使用 `release_label=transfer-final-2026-03-22`
  - within-source 来源使用 `release_label=within-source-final-2026-03-22`
- 完成元数据路径收口与封版校验脚本化：
  - `scripts/normalize_repo_metadata_paths.py` 用于把 `data/models/**`、`data/processed/**` 中仍残留的仓库绝对路径统一改为仓库相对路径
  - `scripts/validate_release_assets.py` 用于固定检查 transfer / multi-seed / ablation / comparison / final release，并做 8 个 `source/model` 的 release resolve smoke
- 保持现有公共接口不变：
  - 生命周期推理仍通过 `/api/v2/predict/lifecycle`
  - 机理解释仍通过 `/api/v2/explain/mechanism`
  - 模型选择继续通过 `model_name=hybrid|bilstm`
  - 每个 `source/model` 都有独立 `final_release.json`，没有引入新的全局默认模型配置

## 当前论文门槛状态

- 真值来源固定为 `*_multi_seed_summary.json`、`*_transfer_summary.json`、`comparison_summary.json`、`ablation_summary.json` 与 `Doc/BHMS论文证据包.md/.json`
- 当前 7 个核心单元里，已通过 5 个，未通过 2 个：
  - `CALCE / within-source`
    - Hybrid `RMSE=0.021945`、`R²=0.513922`
    - BiLSTM `RMSE=0.017631`、`R²=0.695668`
    - 当前 winner：`bilstm`
  - `Kaggle / within-source`
    - Hybrid `RMSE=0.027337`、`R²=0.322002`
    - BiLSTM `RMSE=0.024020`、`R²=0.512857`
    - 当前 winner：`bilstm`
- transfer 主线状态：
  - `CALCE / transfer`：Hybrid 通过，可作为正向证据
  - `NASA / transfer`：Hybrid 通过，但绝对指标仍弱，更适合作为“链路打通 + 弱泛化案例”
- Hybrid 消融门槛当前未通过的来源：
  - `NASA`
  - `CALCE`
  - `MATR`
- 因此，当前更准确的口径是：
  - 系统与实验资产已经封版
  - 论文证据包已经统一
  - “Hybrid 全面优于 BiLSTM”仍未达成，下一阶段必须先重跑并优化 `CALCE / Kaggle within-source` 与 `NASA / CALCE / MATR` 相关消融

## 关键 benchmark 结果

### 1. `CALCE` transfer：`hybrid` 优于 `bilstm`

- `CALCE hybrid transfer`
  - 来源：`data/models/calce/hybrid/transfer/multisource_to_calce/hybrid_transfer_summary.json`
  - `RMSE=0.018598`
  - `R²=0.657456`
- `CALCE bilstm transfer`
  - 来源：`data/models/calce/bilstm/transfer/multisource_to_calce/bilstm_transfer_summary.json`
  - `RMSE=0.024387`
  - `R²=0.412059`
- 结论：在本轮真实 transfer benchmark 中，`CALCE hybrid transfer` 明显优于 `CALCE bilstm transfer`，可作为主展示 transfer 结果。

### 2. `NASA` transfer：已跑通，但泛化结论仍弱

- `NASA hybrid transfer`
  - 来源：`data/models/nasa/hybrid/transfer/multisource_to_nasa/hybrid_transfer_summary.json`
  - `RMSE=0.303572`
  - `R²=-1.117170`
- `NASA bilstm transfer`
  - 来源：`data/models/nasa/bilstm/transfer/multisource_to_nasa/bilstm_transfer_summary.json`
  - `RMSE=0.304276`
  - `R²=-1.134913`
- 结论：`NASA hybrid / bilstm transfer` 都已经真实跑通并形成正式 summary，但当前指标仍弱，不能夸大为“跨源泛化已经充分成立”；更准确的表述应为“transfer 链路已打通，NASA 结果可作为负例或弱泛化案例展示”。

### 3. `HUST / MATR` within-source：扩展验证面已补齐

- `HUST hybrid multi-seed`
  - 来源：`data/models/hust/hybrid/hybrid_multi_seed_summary.json`
  - `RMSE=0.021067`
  - `R²=0.901978`
- `HUST bilstm multi-seed`
  - 来源：`data/models/hust/bilstm/bilstm_multi_seed_summary.json`
  - `RMSE=0.021182`
  - `R²=0.900922`
- `MATR hybrid multi-seed`
  - 来源：`data/models/matr/hybrid/hybrid_multi_seed_summary.json`
  - `RMSE=0.098888`
  - `R²=0.294000`
- `MATR bilstm multi-seed`
  - 来源：`data/models/matr/bilstm/bilstm_multi_seed_summary.json`
  - `RMSE=0.097878`
  - `R²=0.308747`
- 结论：`HUST / MATR` 现已具备完整 within-source 展示面，可作为扩展验证源；其中 `HUST` 结果较强，`MATR` 结果中等，适合与 `CALCE / NASA` 分层展示而不是混写成统一强结论。

## final release 覆盖范围

### 已纳入 final release

- `CALCE / hybrid`：`data/models/calce/hybrid/release/final_release.json`
- `CALCE / bilstm`：`data/models/calce/bilstm/release/final_release.json`
- `NASA / hybrid`：`data/models/nasa/hybrid/release/final_release.json`
- `NASA / bilstm`：`data/models/nasa/bilstm/release/final_release.json`
- `HUST / hybrid`：`data/models/hust/hybrid/release/final_release.json`
- `HUST / bilstm`：`data/models/hust/bilstm/release/final_release.json`
- `MATR / hybrid`：`data/models/matr/hybrid/release/final_release.json`
- `MATR / bilstm`：`data/models/matr/bilstm/release/final_release.json`

### 未纳入 final release

- `Kaggle`
  - 当前保留为基线/对照资产，不进入本轮 final release 集。
- `Oxford`
  - 当前仍作为辅助 trajectory 源，不进入本轮 lifecycle final release 集。
- `PulseBat`
  - 当前仍作为 enhancement-only 机制源，不进入本轮 lifecycle final release 集。

## 已知限制与风险

- 本轮已经完成 lifecycle-first 工程封版资产收口，但这不等同于“所有来源效果都已达到论文级最优”。
- 当前论文门槛尚未通过；在 `python scripts/validate_release_assets.py --require-paper-gate` 通过前，不应在论文主结论中写“Hybrid 在核心 benchmark 上全面优于 BiLSTM”。
- `NASA` transfer 已经真实跑通，但指标仍弱；因此当前更适合将其作为 transfer 链路证明和弱泛化案例，而不是强结论主证据。
- 正式 release checkpoint 已收口到 `data/models/<source>/<model>/release/checkpoints/`；中间训练材料应统一归档到 `data/archive/experiments/<date>/`，不再混留在主工作区。
- 当前系统定位仍是毕业设计/研究原型的工程封版，不应写成“工业级部署完成”，也不应写成“工业级可靠性验证已完成”。

## 建议的下一步

- 提交准备阶段：优先围绕 `README.md`、`Doc/BHMS封版检查清单.md`、`Doc/BHMS最终封版说明.md` 与正式 summary / release manifest 组织提交说明，形成一套可复核的封版提交包。
- 归档阶段：执行 `python scripts/archive_experiment_artifacts.py --archive-label <date>`，将 `runs/`、per-seed checkpoint、`training_summary.json`、`test_details.json` 等中间训练材料统一转入 `data/archive/experiments/<date>/`。
- 重跑阶段：优先重做 `CALCE / MATR within-source` 的 Hybrid 与 BiLSTM 公平对比，再重新执行对应 `ablation` 与 `python scripts/rebuild_benchmark_truth.py`。
- 答辩/汇报阶段：主口径建议以 `CALCE` transfer 为主展示、`NASA` transfer 为限制说明、`HUST / MATR` 为当前正向 within-source 支撑，`CALCE / Kaggle within-source` 作为“仍在优化”的局限性说明。
