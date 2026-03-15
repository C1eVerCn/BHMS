"""成品级数据准备、预测与诊断流程测试。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.database import DatabaseManager  # noqa: E402
from backend.app.services.battery_service import BatteryService  # noqa: E402
from backend.app.services.insight_service import InsightService  # noqa: E402
from backend.app.services.model_service import PredictionService  # noqa: E402
from backend.app.services.repository import BHMSRepository  # noqa: E402
from backend.app.services.training_service import TrainingService  # noqa: E402
from ml.data.schema import finalize_cycle_frame  # noqa: E402


def _make_settings(tmp_path: Path):
    base = get_settings()
    data_dir = tmp_path / 'data'
    return replace(
        base,
        project_root=tmp_path,
        data_dir=data_dir,
        raw_nasa_dir=data_dir / 'raw' / 'nasa',
        raw_calce_dir=data_dir / 'raw' / 'calce',
        raw_kaggle_dir=data_dir / 'raw' / 'kaggle',
        processed_dir=data_dir / 'processed',
        knowledge_path=PROJECT_ROOT / 'data' / 'knowledge' / 'battery_fault_knowledge.json',
        model_dir=PROJECT_ROOT / 'data' / 'models',
        upload_dir=data_dir / 'uploads',
        demo_upload_dir=data_dir / 'demo_uploads',
        database_path=data_dir / 'bhms.db',
        graph_backend='memory',
    )


def _build_cycle_frame(source: str, battery_prefix: str, battery_count: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery_index in range(battery_count):
        for cycle in range(1, 26):
            rows.append(
                {
                    'source_battery_id': f'{battery_prefix}_{battery_index:03d}',
                    'cycle_number': cycle,
                    'voltage_mean': 3.72 - cycle * 0.002,
                    'current_mean': -1.85,
                    'temperature_mean': 24.5 + cycle * 0.08,
                    'temperature_rise_rate': 0.5 + cycle * 0.03,
                    'internal_resistance': 0.02 + cycle * 0.0004,
                    'capacity': 2.0 - cycle * 0.018 - battery_index * 0.01,
                }
            )
    return finalize_cycle_frame(pd.DataFrame(rows), source=source, dataset_name=f'{source}_upload_test', eol_capacity_ratio=0.8)


def test_prepare_training_dataset_uses_training_pool(tmp_path: Path):
    settings = _make_settings(tmp_path)
    database = DatabaseManager(settings.database_path)
    database.initialize()
    repo = BHMSRepository(database)
    service = BatteryService(repository=repo, settings=settings)

    frame = _build_cycle_frame('calce', 'CALCE_POOL')
    csv_path = tmp_path / 'pool.csv'
    frame.to_csv(csv_path, index=False)
    service.import_frame(frame, source='calce', dataset_path=csv_path, include_in_training=True)

    payload = service.prepare_training_dataset('calce', seq_len=12, batch_size=4)
    assert Path(payload['csv_path']).exists()
    assert payload['data_summary']['num_batteries'] == 4
    assert payload['import_summary']['validation_summary']['include_in_training'] is True


def test_prediction_and_diagnosis_return_explainable_payloads(tmp_path: Path):
    settings = _make_settings(tmp_path)
    database = DatabaseManager(settings.database_path)
    database.initialize()
    repo = BHMSRepository(database)
    battery_service = BatteryService(repository=repo, settings=settings)
    prediction_service = PredictionService(repository=repo, settings=settings)

    frame = _build_cycle_frame('kaggle', 'KAGGLE_FLOW', battery_count=3)
    csv_path = tmp_path / 'predict.csv'
    frame.to_csv(csv_path, index=False)
    summary = battery_service.import_frame(frame, source='kaggle', dataset_path=csv_path, include_in_training=True)
    battery_id = summary['battery_ids'][0]

    prediction = prediction_service.predict_rul(battery_id=battery_id, model_name='hybrid', seq_len=20)
    assert prediction['projection']['forecast_points']
    assert prediction['explanation']['feature_contributions']
    assert '电池寿命预测报告' in prediction['report_markdown']

    diagnosis = prediction_service.diagnose(
        battery_id=battery_id,
        anomalies=[{'symptom': '温度异常', 'severity': 'high', 'description': '平均温度升高明显', 'code': 'temperature_anomaly'}],
        battery_info={'battery_id': battery_id},
    )
    assert diagnosis['candidate_faults']
    assert diagnosis['graph_trace']['nodes']
    assert '电池故障诊断报告' in diagnosis['report_markdown']


def test_training_overview_aggregates_experiment_assets(tmp_path: Path):
    settings = _make_settings(tmp_path)
    database = DatabaseManager(settings.database_path)
    database.initialize()
    repo = BHMSRepository(database)
    battery_service = BatteryService(repository=repo, settings=settings)
    training_service = TrainingService(repository=repo, settings=settings)

    frame = _build_cycle_frame('calce', 'CALCE_OVERVIEW')
    csv_path = tmp_path / 'overview.csv'
    frame.to_csv(csv_path, index=False)
    battery_service.import_frame(frame, source='calce', dataset_path=csv_path, include_in_training=True)
    battery_service.prepare_training_dataset('calce', seq_len=12, batch_size=4)

    overview = training_service.get_overview()
    detail = training_service.get_experiment_detail('calce')
    ablation = training_service.get_ablation_summary('calce')

    assert len(overview['sources']) == 3
    calce_overview = next(item for item in overview['sources'] if item['source'] == 'calce')
    assert calce_overview['dataset_batteries'] == 4
    assert detail['dataset_summary']['num_batteries'] == 4
    assert detail['comparison']['source'] == 'calce'
    assert {'multi_seed_hybrid', 'multi_seed_bilstm', 'ablation_study'} <= set(detail['recommended_commands'])
    assert ablation['source'] == 'calce'
    assert isinstance(ablation['variants'], list) and ablation['variants']


def test_insight_service_exposes_profile_case_bundle_and_system_status(tmp_path: Path):
    settings = _make_settings(tmp_path)
    database = DatabaseManager(settings.database_path)
    database.initialize()
    repo = BHMSRepository(database)
    battery_service = BatteryService(repository=repo, settings=settings)
    prediction_service = PredictionService(repository=repo, settings=settings)
    insight_service = InsightService(repository=repo, settings=settings)

    frame = _build_cycle_frame('calce', 'CALCE_CASE', battery_count=3)
    csv_path = tmp_path / 'case.csv'
    frame.to_csv(csv_path, index=False)
    summary = battery_service.import_frame(frame, source='calce', dataset_path=csv_path, include_in_training=True)
    battery_service.prepare_training_dataset('calce', seq_len=12, batch_size=4)
    battery_id = summary['battery_ids'][0]

    prediction_service.predict_rul(battery_id=battery_id, model_name='hybrid', seq_len=20)
    prediction_service.diagnose(
        battery_id=battery_id,
        anomalies=[{'symptom': '温度异常', 'severity': 'high', 'description': '平均温度升高明显', 'code': 'temperature_anomaly'}],
        battery_info={'battery_id': battery_id},
    )

    profile = insight_service.get_dataset_profile('calce')
    bundle = insight_service.get_case_bundle(battery_id)
    status = insight_service.get_system_status()

    assert profile['battery_count'] == 3
    assert profile['training_candidate_count'] == 3
    assert profile['comparison_available'] is True
    assert bundle['battery_id'] == battery_id
    assert bundle['prediction'] is not None
    assert bundle['diagnosis'] is not None
    assert 'BHMS 案例包' in bundle['bundle_markdown']
    assert status['database_ready'] is True
    assert '导出预测/诊断报告' in status['demo_acceptance_flow']
