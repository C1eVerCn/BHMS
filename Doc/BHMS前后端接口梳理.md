# BHMS 前后端接口梳理

## 1. 当前系统主链路

BHMS 已经进入 `lifecycle-first` 形态，当前主链路是：

`数据导入 -> 生命周期预测 -> 异常检测 -> GraphRAG 机理解释 -> 案例导出`

对应的代码入口：

- 前端应用入口：`frontend/src/App.tsx`
- 前端状态层：`frontend/src/stores/useBhmsStore.ts`
- 前端接口层：`frontend/src/services/bhms.ts`
- 后端应用入口：`backend/app/main.py`
- 后端聚合路由：`backend/app/api/router.py`
- 生命周期/诊断服务：`backend/app/services/model_service.py`
- 数据导入与电池服务：`backend/app/services/battery_service.py`
- SQLite 仓储层：`backend/app/services/repository.py`
- 生命周期推理入口：`ml/inference/predictor.py`

## 2. 前端页面与后端接口对应关系

| 前端页面 | 状态动作 | 后端接口 | 后端服务 |
| --- | --- | --- | --- |
| `Dashboard.tsx` | `loadDashboard` | `GET /api/v1/dashboard/summary` | `BatteryService.get_dashboard` |
| `BatteryList.tsx` | `loadBatteries` / `loadBatteryContext` | `GET /api/v1/batteries` / `GET /api/v1/battery/{battery_id}` 等 | `BatteryService` |
| `DataUpload.tsx` | `uploadFile` / `importSource` / `importPreset` | `POST /api/v1/data/upload` / `POST /api/v1/data/import-source` / `POST /api/v1/data/import-demo-preset` | `BatteryService` |
| `Prediction.tsx` | `runLifecyclePrediction` | `POST /api/v2/predict/lifecycle` | `PredictionService.predict_lifecycle` |
| `Diagnosis.tsx` | `runDiagnosisWorkflow` | `POST /api/v2/predict/lifecycle` + `POST /api/v1/detect/anomaly` + `POST /api/v2/explain/mechanism` | `PredictionService` |
| `Analysis.tsx` | `loadCaseBundle` / `exportCaseBundle` | `GET /api/v1/reports/case-bundle/{battery_id}` / `POST /api/v1/reports/case-bundle/{battery_id}/export` | `InsightService` |

## 3. 生命周期相关接口

### 3.1 生命周期预测

- 路径：`POST /api/v2/predict/lifecycle`
- 请求体：

```json
{
  "battery_id": "calce::sample::001",
  "model_name": "hybrid",
  "seq_len": 30,
  "historical_data": []
}
```

- 关键字段说明：
  - `battery_id`：当前样本 ID
  - `model_name`：当前支持 `hybrid | bilstm | auto`
  - `seq_len`：用于推理的历史窗口长度
  - `historical_data`：可选；不传时从 SQLite 历史周期点读取

- 返回核心字段：
  - `model_name` / `model_version` / `checkpoint_id`
  - `predicted_rul`
  - `predicted_knee_cycle`
  - `predicted_eol_cycle`
  - `trajectory`
  - `projection`
  - `risk_windows`
  - `future_risks`
  - `model_evidence`
  - `fallback_used`

### 3.2 机理解释

- 路径：`POST /api/v2/explain/mechanism`
- 当前约定：该接口现在显式接收 `model_name` 与 `seq_len`，并在缺少对应生命周期预测时，先自动补齐一次生命周期预测，再进入 GraphRAG 诊断。
- 请求体：

```json
{
  "battery_id": "calce::sample::001",
  "model_name": "bilstm",
  "seq_len": 30,
  "anomalies": [],
  "battery_info": {}
}
```

- 返回核心字段：
  - `fault_type` / `confidence` / `severity`
  - `candidate_faults`
  - `graph_trace`
  - `decision_basis`
  - `lifecycle_evidence`
  - `model_evidence`
  - `graph_backend`

## 4. 后端接口分组清单

### 4.1 数据与样本

- `GET /api/v1/batteries`
- `GET /api/v1/batteries/options`
- `GET /api/v1/battery/{battery_id}`
- `GET /api/v1/battery/{battery_id}/cycles`
- `GET /api/v1/battery/{battery_id}/history`
- `GET /api/v1/battery/{battery_id}/health`
- `POST /api/v1/battery/{battery_id}/training-candidate`
- `POST /api/v1/data/upload`
- `POST /api/v1/data/import-source`
- `POST /api/v1/data/import-demo-preset`

### 4.2 生命周期与诊断

- `POST /api/v2/predict/lifecycle`
- `POST /api/v1/detect/anomaly`
- `POST /api/v2/explain/mechanism`

### 4.3 训练与实验分析

- `GET /api/v1/training/comparison`
- `GET /api/v1/training/overview`
- `GET /api/v1/training/experiments/{source}`
- `GET /api/v1/training/ablations/{source}`

### 4.4 分析与导出

- `GET /api/v1/system/status`
- `GET /api/v1/demo/presets`
- `GET /api/v1/data/profile/{source}`
- `GET /api/v1/diagnosis/knowledge/summary`
- `GET /api/v1/reports/case-bundle/{battery_id}`（已内含 `prediction.report_markdown` 与 `diagnosis.report_markdown`）
- `POST /api/v1/reports/case-bundle/{battery_id}/export`

## 5. 模型调用约定

### 5.1 当前正式支持的生命周期模型

- `hybrid`：`LifecycleHybridPredictor`
- `bilstm`：`LifecycleBiLSTMPredictor`

### 5.2 权重解析顺序

`ml/inference/predictor.py` 当前解析顺序：

1. `data/models/<source>/<model>/release/final_release.json`
2. transfer summary
3. multi-seed summary
4. 单次 checkpoint
5. 若全部缺失，退回 heuristic

### 5.3 当前前后端统一约定

- 前端预测页、诊断页共用同一份 `lifecycleRequestConfig`
- `model_name` 与 `seq_len` 由 Zustand 统一保存
- 诊断工作流在进入 `explain/mechanism` 前，先按当前模型配置补齐生命周期预测
- 上传页“一键进入完整分析链路”会复用当前模型配置，而不是写死 `hybrid`

## 6. 当前保留与清理边界

- 生命周期闭环以 `POST /api/v2/predict/lifecycle` 为主接口
- 旧的 `POST /api/v1/predict/rul` 与 `POST /api/v1/diagnose` 已移除，避免继续维持双接口链路
- 前端不再维护旧的 `runPrediction` 别名调用链，统一走 `runLifecyclePrediction`
- 诊断链路不再依赖“历史上恰好存在一条预测记录”这一隐式前提

## 7. 建议的后续整理顺序

1. 如果继续做接口收口，优先补 `FastAPI response_model`
2. 若要支持案例导出时切换模型，可再给 `case-bundle/export` 增加 `model_name` 参数
3. 若要做前端对比视图，可在分析页增加 `hybrid vs bilstm` 并排对照面板
