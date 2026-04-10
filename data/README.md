# BHMS 数据目录说明

本文档以当前仓库真实状态为准，说明 `data/` 目录的结构、数据获取方式和恢复校验命令。

## 当前目录结构

```text
data/
├── raw/                         # 原始数据与外部下载产物
│   ├── nasa/
│   ├── calce/
│   ├── kaggle/
│   ├── hust/
│   ├── matr/
│   ├── oxford/
│   └── pulsebat/
├── processed/                   # 训练/推理使用的结构化数据
├── models/                      # 训练产物、multi-seed summary、release manifest
├── exports/cases/               # 运行时导出的案例目录，可按需清空后重新生成
├── archive/                     # ignored 的历史实验归档与恢复备份
├── knowledge/                   # 机理知识与图谱静态资源
├── demo_uploads/                # 演示样本
└── uploads/                     # 运行时上传数据
```

## 现役数据源

BHMS 当前已接入并验证的现役原始数据源如下：

- `nasa`: NASA PCoE Battery Dataset，MAT 文件，训练与测试都会直接使用。
- `calce`: CALCE 演示/基线数据，已随仓库维护。
- `kaggle`: Kaggle 演示/基线数据，仅在本机已有凭证时补齐。
- `hust`: HUST 77-cell 生命周期数据，原始发布包为 `hust_data.zip`。
- `matr`: MATR 原始 `MAT` 批次文件。
- `oxford`: Oxford Battery Degradation Dataset 1，原始 `MAT` 文件。
- `pulsebat`: PulseBat 上游仓库原始 ZIP 与解压内容。

## 数据获取脚本

请使用当前仓库存在的脚本，不要再使用旧文档里的 `download_nasa_data_final.py`、`verify_data.py` 等已不存在脚本。

### NASA

```bash
./.venv/bin/python scripts/download_nasa_data.py --download
./.venv/bin/python scripts/download_nasa_data.py --verify
```

### HUST / MATR

`download_hust_matr.py` 采用分片下载；当前版本会先校验本地文件，已验证完整的文件会直接跳过。

```bash
./.venv/bin/python scripts/download_hust_matr.py
```

### Oxford / PulseBat / HUST / MATR 状态记录

`acquire_external_datasets.py` 会下载或跳过现有完整文件，并重写 `DATASET_STATUS.md` / `dataset_status.json`。

```bash
./.venv/bin/python scripts/acquire_external_datasets.py --source oxford
./.venv/bin/python scripts/acquire_external_datasets.py --source pulsebat
./.venv/bin/python scripts/acquire_external_datasets.py --source hust
./.venv/bin/python scripts/acquire_external_datasets.py --source matr
```

### Kaggle

仅在本机已配置 Kaggle 凭证时使用：

```bash
./.venv/bin/python scripts/download_kaggle_data.py --help
```

## release 资产

正式推理资产位于：

```text
data/models/<source>/<model>/release/
├── checkpoints/<model>_release.pt
└── final_release.json
```

multi-seed / transfer summary 会把 `best_checkpoint.path` 指向 release 目录中的正式 checkpoint。

重新晋升 release 的入口脚本：

```bash
./.venv/bin/python scripts/promote_lifecycle_release.py --source hust --model hybrid
./.venv/bin/python scripts/promote_lifecycle_release.py --source matr --model bilstm
```

## 恢复与校验命令

### 1. 检查现役 raw / formal release 下是否还有 `dataless`

```bash
find data/raw data/models -type f -exec ls -lO {} + | rg dataless
```

### 2. 校验 release manifest 与 summary

```bash
./.venv/bin/python scripts/validate_release_assets.py
```

### 3. 运行 Python 回归测试

```bash
./.venv/bin/python -m pytest -q
```

### 4. 构建前端

```bash
cd frontend
npm run build -- --outDir /tmp/bhms-frontend-build-$(date +%Y%m%d)
```

## 训练环境说明

当前仓库已将训练栈锁定在 `requirements.txt` 中的已验证组合，核心版本为：

- `torch==2.5.1`
- `numpy==1.26.4`
- `scipy==1.11.4`
- `pandas==2.2.3`
- `scikit-learn==1.5.2`
- `h5py==3.12.1`

如需重建环境，建议直接在项目根目录重新安装：

```bash
./.venv/bin/python -m pip install --upgrade --force-reinstall -r requirements.txt
```

## 案例导出说明

`data/exports/cases/` 只保存运行时生成的案例包，不再作为正式封版资产长期留仓。

- 当前目录可以整体清空，系统会在下一次执行案例导出时自动重建
- 正式可复核资产应优先查看 `data/models/**` 下的 release / summary 与 `Doc/BHMS论文证据包.md`
