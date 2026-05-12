# BHMS 数据集管理指南

本文档详细说明如何下载、组织和使用BHMS项目所需的数据集。

## 📁 目录结构

```
data/
├── raw/                    # 原始数据
│   ├── nasa/              # NASA PCoE数据集
│   ├── kaggle/            # Kaggle数据集
│   └── calce/             # CALCE数据集
├── processed/             # 预处理后的数据
│   ├── nasa/             # NASA数据处理后
│   └── features/         # 提取的特征
├── features/              # 特征工程输出
│   ├── train/            # 训练集特征
│   ├── val/              # 验证集特征
│   └── test/             # 测试集特征
├── models/                # 训练好的模型权重
└── knowledge/             # 知识图谱数据
```

## 🚀 快速开始

### 1. 一键下载所有数据

```bash
# 下载NASA数据
python scripts/download_nasa_data.py --all

# 下载Kaggle数据（需要先配置Kaggle API）
python scripts/download_kaggle_data.py --all
```

### 2. 验证数据完整性

```bash
python scripts/verify_data.py
```

## 📊 数据集详情

### NASA PCoE Battery Dataset

**数据提供方**: NASA Prognostics Center of Excellence  
**官方网址**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/  
**AWS S3链接**: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip  
**授权情况**: 公开数据集，可用于学术研究  
**数据格式**: MATLAB (.mat)

#### 包含的数据文件

| 电池ID | 类型 | 循环次数 | 测试条件 | 文件大小 |
|--------|------|----------|----------|----------|
| B0005 | 18650 Li-ion | 168 | 室温(24°C), 2A放电 | ~15MB |
| B0006 | 18650 Li-ion | 168 | 室温(24°C), 2A放电 | ~15MB |
| B0007 | 18650 Li-ion | 168 | 室温(24°C), 2A放电 | ~15MB |
| B0018 | 18500 Li-ion | 132 | 室温(24°C), 2A放电 | ~12MB |
| B0029 | 18650 Li-ion | 104 | 加速老化, 4A放电 | ~10MB |
| B0030 | 18650 Li-ion | 120 | 加速老化, 4A放电 | ~11MB |
| B0031 | 18650 Li-ion | 156 | 加速老化, 4A放电 | ~14MB |
| B0032 | 18650 Li-ion | 140 | 加速老化, 4A放电 | ~13MB |

#### 数据字段说明

- `Voltage_measured`: 测量电压 (V)
- `Current_measured`: 测量电流 (A)
- `Temperature_measured`: 测量温度 (°C)
- `Current_load`: 负载电流 (A)
- `Voltage_load`: 负载电压 (V)
- `Time`: 时间戳
- `Capacity`: 放电容量 (Ah)

---

### Kaggle Battery Datasets

**平台**: Kaggle (https://www.kaggle.com)  
**授权情况**: 按各数据集具体授权  
**数据格式**: CSV, MAT

#### 推荐数据集

1. **patrickfleith/lithium-ion-battery-aging-datasets**
   - 描述: 锂电池老化数据集合集
   - 大小: ~500MB
   - 包含: NASA、CALCE等多个数据集的整合

2. **sahilwagh/lithium-ion-battery-aging-dataset**
   - 描述: NASA和CALCE数据集整合
   - 大小: ~200MB
   - 格式: CSV

3. **vinayakshanawad/lithium-ion-battery-life-prediction**
   - 描述: 电池寿命预测数据集
   - 大小: ~100MB
   - 适合: RUL预测任务

---

## 🔧 详细下载步骤

### 方法一: 使用自动化脚本（推荐）

#### NASA PCoE 数据下载

**注意**: NASA数据现在托管在AWS S3上，使用以下链接:
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

```bash
# 查看可用数据集
python scripts/download_nasa_data_final.py --list

# 下载并解压所有数据（推荐）
python scripts/download_nasa_data_final.py --download

# 验证数据完整性
python scripts/download_nasa_data_final.py --verify

# 指定保存目录
python scripts/download_nasa_data_final.py --download --save-dir ./my_data/nasa
```

#### Kaggle 数据下载

```bash
# 首次使用，查看设置指南
python scripts/download_kaggle_data.py --setup

# 列出推荐数据集
python scripts/download_kaggle_data.py --list

# 下载所有推荐数据集
python scripts/download_kaggle_data.py --all

# 下载指定数据集
python scripts/download_kaggle_data.py --dataset patrickfleith

# 搜索其他数据集
python scripts/download_kaggle_data.py --search "battery aging"
```

---

### 方法二: 手动下载

#### NASA PCoE 手动下载步骤

1. **访问官方网站**
   ```
   https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
   ```

2. **下载数据文件**
   - 找到 "Battery Data Set" 部分
   - 下载以下文件:
     - B0005.mat
     - B0006.mat
     - B0007.mat
     - B0018.mat
     - B0029.mat
     - B0030.mat
     - B0031.mat
     - B0032.mat

3. **移动到项目目录**
   ```bash
   # macOS/Linux
   mv ~/Downloads/B00*.mat data/raw/nasa/

   # Windows PowerShell
   Move-Item $env:USERPROFILE\Downloads\B00*.mat data\raw\nasa\
   ```

#### Kaggle 手动下载步骤

1. **注册Kaggle账号**
   - 访问 https://www.kaggle.com
   - 点击 "Sign Up" 注册
   - 验证邮箱

2. **下载数据**
   - 访问数据集页面（如：https://www.kaggle.com/patrickfleith/lithium-ion-battery-aging-datasets）
   - 点击 "Download" 按钮
   - 解压下载的zip文件

3. **移动到项目目录**
   ```bash
   unzip lithium-ion-battery-aging-datasets.zip -d data/raw/kaggle/
   ```

---

## 🔐 Kaggle API 配置详解

### 步骤 1: 安装Kaggle API

```bash
pip install kaggle
```

### 步骤 2: 获取API Token

1. 登录 https://www.kaggle.com
2. 点击右上角头像 → "Account"
3. 向下滚动到 "API" 部分
4. 点击 "Create New Token"
5. 下载 `kaggle.json` 文件

### 步骤 3: 配置凭证（按操作系统）

#### macOS

```bash
# 创建配置目录
mkdir -p ~/.kaggle

# 移动凭证文件
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置权限（重要！）
chmod 600 ~/.kaggle/kaggle.json

# 验证
kaggle --version
```

#### Linux

```bash
# 创建配置目录
mkdir -p ~/.kaggle

# 移动凭证文件
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置权限
chmod 600 ~/.kaggle/kaggle.json

# 验证
kaggle --version
```

#### Windows

```powershell
# 创建配置目录
mkdir %USERPROFILE%\.kaggle

# 移动凭证文件
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# 设置文件为只读（可选但推荐）
attrib +R %USERPROFILE%\.kaggle\kaggle.json

# 验证
kaggle --version
```

### 步骤 4: 测试API

```bash
# 查看Kaggle版本
kaggle --version

# 列出数据集
kaggle datasets list

# 搜索电池数据集
kaggle datasets list -s "lithium battery"
```

---

## 📦 数据预处理

### 加载NASA数据

```python
from scipy.io import loadmat
import pandas as pd

# 加载MAT文件
data = loadmat('data/raw/nasa/B0005.mat')

# 提取电池数据
battery_data = data['B0005'][0][0][0][0]

# 转换为DataFrame
cycles = []
for i, cycle in enumerate(battery_data):
    cycle_data = {
        'cycle_number': i + 1,
        'capacity': cycle[0][0][0][0][0] if cycle[0][0][0].size > 0 else None,
        'voltage': cycle[0][0][1].flatten(),
        'current': cycle[0][0][2].flatten(),
        'temperature': cycle[0][0][3].flatten(),
    }
    cycles.append(cycle_data)

df = pd.DataFrame(cycles)
print(df.head())
```

### 数据验证

```python
import os
from pathlib import Path

def verify_nasa_data():
    """验证NASA数据完整性"""
    nasa_dir = Path('data/raw/nasa')
    
    expected_files = [
        'B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat',
        'B0029.mat', 'B0030.mat', 'B0031.mat', 'B0032.mat'
    ]
    
    missing = []
    for f in expected_files:
        if not (nasa_dir / f).exists():
            missing.append(f)
    
    if missing:
        print(f"❌ 缺失文件: {missing}")
        return False
    else:
        print("✅ 所有NASA数据文件已就位")
        return True

verify_nasa_data()
```

---

## 🛠️ 常见问题解决

### 问题 1: NASA下载失败（403 Forbidden）

**原因**: NASA网站可能需要特定的请求头  
**解决**: 使用脚本中的下载方式，已配置User-Agent

```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
```

### 问题 2: Kaggle API 401 Unauthorized

**原因**: API凭证未正确配置  
**解决**: 
1. 确认 `kaggle.json` 文件存在
2. 检查文件权限（macOS/Linux需要600）
3. 重新下载Token

### 问题 3: 找不到kaggle命令

**原因**: Python脚本目录未加入PATH  
**解决**:

```bash
# 找到kaggle安装位置
which kaggle
# 或
pip show kaggle

# 添加到PATH（临时）
export PATH="$PATH:$HOME/.local/bin"

# 永久添加（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
source ~/.zshrc
```

### 问题 4: 数据文件损坏

**验证文件完整性**:

```bash
# 查看文件大小
ls -lh data/raw/nasa/

# 使用Python验证MAT文件
python -c "from scipy.io import loadmat; loadmat('data/raw/nasa/B0005.mat'); print('OK')"
```

---

## 📋 数据使用规范

### 引用要求

使用NASA PCoE数据集时，请在论文中引用：

```bibtex
@misc{nasa_battery_dataset,
  title = {Prognostics Center of Excellence - Battery Data Set},
  author = {{NASA Ames Research Center}},
  year = {2008},
  howpublished = {\url{https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/}}
}
```

### 数据隐私

- 所有数据集均为公开数据集
- 不包含个人隐私信息
- 可用于学术研究和商业应用（需遵守各数据集授权）

---

## 🔗 相关链接

- [NASA PCoE Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [CALCE Battery Data](https://calce.umd.edu/battery-data)
- [Project GitHub](https://github.com/your-org/bhms)

---

## 📞 技术支持

如有数据下载或使用问题，请：

1. 查看本文档的"常见问题"部分
2. 运行验证脚本: `python scripts/verify_data.py`
3. 提交Issue到项目仓库

---

*最后更新: 2026-03-11*
