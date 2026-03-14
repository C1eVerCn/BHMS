"""数据处理模块。"""

from ml.data.nasa_preprocessor import DEFAULT_FEATURE_COLUMNS, DatasetSplit, NASABatteryPreprocessor

__all__ = ["DEFAULT_FEATURE_COLUMNS", "DatasetSplit", "NASABatteryPreprocessor"]

try:
    from ml.data.dataset import BatterySequenceDataset, NormalizationStats, RULDataModule, create_synthetic_data

    __all__.extend(["BatterySequenceDataset", "NormalizationStats", "RULDataModule", "create_synthetic_data"])
except ModuleNotFoundError:
    # 允许在未安装 torch 的推理/后端环境中仅使用预处理能力。
    pass
