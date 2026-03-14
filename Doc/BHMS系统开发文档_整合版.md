# 锂电池健康管理系统（BHMS）开发文档

## 文档控制信息

| 属性 | 内容 |
|-----|------|
| 文档名称 | 锂电池健康管理系统（BHMS）开发文档 |
| 文档版本 | V2.0（整合版） |
| 编写日期 | 2026年3月10日 |
| 最后更新 | 2026年3月10日 |
| 文档状态 | 正式发布 |
| 编写人员 | BHMS项目团队 |
| 审核人员 | 技术负责人 |
| 密级 | 内部公开 |

### 版本历史

| 版本 | 日期 | 修订人 | 修订内容 |
|-----|------|-------|---------|
| V1.0 | 2026-03-10 | 项目团队 | 初稿创建 |
| V2.0 | 2026-03-10 | 项目团队 | 整合V1和V2版本，补充完整开发流程 |

---

## 目录

1. [开发环境配置](#1-开发环境配置)
2. [项目结构说明](#2-项目结构说明)
3. [开发规范](#3-开发规范)
4. [模块实现指南](#4-模块实现指南)
5. [模型训练完整流程](#5-模型训练完整流程)
6. [错误处理策略](#6-错误处理策略)
7. [前端状态管理](#7-前端状态管理)
8. [API开发指南](#8-api开发指南)
9. [测试策略](#9-测试策略)
10. [CI/CD流程](#10-cicd流程)
11. [部署指南](#11-部署指南)
12. [监控与运维](#12-监控与运维)
13. [附录](#13-附录)

---

## 1. 开发环境配置

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 | 说明 |
|-----|---------|---------|------|
| CPU | 4核 | 8核+ | 模型训练需要多核CPU |
| 内存 | 16GB | 32GB+ | 大数据集处理 |
| 存储 | 100GB SSD | 500GB NVMe SSD | 模型文件和数据存储 |
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 4090 24GB | 深度学习训练和推理 |
| 网络 | 100Mbps | 1Gbps | 数据下载和模型部署 |

### 1.2 软件环境

#### 1.2.1 操作系统

- **开发环境**: macOS 13+ / Ubuntu 22.04 LTS / Windows 11 WSL2
- **生产环境**: Ubuntu 22.04 LTS (推荐)

#### 1.2.2 基础依赖

| 软件 | 版本 | 安装命令 |
|-----|------|---------|
| Python | 3.9-3.11 | `pyenv install 3.11.8` |
| Node.js | 18.x LTS | `nvm install 18` |
| Docker | 24.x+ | 官网下载安装包 |
| Docker Compose | 2.20+ | 随Docker安装 |
| Git | 2.40+ | `apt install git` |
| CUDA | 11.8+ | NVIDIA官网下载 |

#### 1.2.3 Python环境初始化

```bash
# 1. 克隆项目
git clone https://github.com/your-org/bhms.git
cd bhms

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖

# 4. 安装pre-commit钩子
pre-commit install
```

**requirements.txt 核心依赖**:
```txt
# Web框架
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# 数据库
sqlalchemy==2.0.23
alembic==1.12.1
neo4j==5.14.0
redis==5.0.1

# 机器学习
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
scipy==1.11.4

# 数据处理
h5py==3.10.0
openpyxl==3.1.2
python-dateutil==2.8.2

# 安全
pyjwt==2.8.0
bcrypt==4.1.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# 工具
pydantic==2.5.0
python-dotenv==1.0.0
httpx==0.25.2
celery==5.3.4
prometheus-client==0.19.0

# 日志
structlog==23.2.0
```

#### 1.2.4 Node.js环境初始化

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev
```

**package.json 核心依赖**:
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "zustand": "^4.4.7",
    "antd": "^5.12.0",
    "@ant-design/icons": "^5.2.6",
    "echarts": "^5.4.3",
    "echarts-for-react": "^3.0.2",
    "axios": "^1.6.2",
    "dayjs": "^1.11.10",
    "lodash-es": "^4.17.21"
  },
  "devDependencies": {
    "@types/react": "^18.2.37",
    "@types/react-dom": "^18.2.15",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.2.2",
    "vite": "^5.0.0",
    "eslint": "^8.53.0",
    "prettier": "^3.1.0"
  }
}
```

#### 1.2.5 Neo4j数据库安装

**Docker方式安装**：

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -v $HOME/neo4j/data:/data \
  -v $HOME/neo4j/logs:/logs \
  neo4j:5.15
```

**本地安装**：

```bash
# macOS
brew install neo4j
brew services start neo4j

# Ubuntu
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j=1:5.15.0
sudo systemctl start neo4j
```

**访问Web控制台**：http://localhost:7474

### 1.3 环境变量配置

**创建 `.env` 文件**:

```bash
# 数据库配置
DATABASE_URL=sqlite:///data/bhms.db
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
REDIS_URL=redis://localhost:6379/0

# 安全配置
SECRET_KEY=your-super-secret-key-min-32-chars-long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# 模型配置
MODEL_PATH=./models
DEFAULT_MODEL_VERSION=v1.0.0
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0

# LLM配置
LLM_PROVIDER=deepseek
LLM_API_KEY=your-api-key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=json

# 监控配置
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

---

## 2. 项目结构说明

### 2.1 目录结构

```
BHMS/
├── docs/                          # 项目文档
│   ├── design/                    # 设计文档
│   ├── api/                       # API文档
│   └── guides/                    # 开发指南
│
├── frontend/                      # 前端项目 (React + TypeScript)
│   ├── public/                    # 静态资源
│   ├── src/
│   │   ├── components/            # 公共组件
│   │   │   ├── common/            # 通用组件
│   │   │   ├── charts/            # 图表组件
│   │   │   └── forms/             # 表单组件
│   │   ├── pages/                 # 页面组件
│   │   │   ├── dashboard/         # 仪表盘
│   │   │   ├── data/              # 数据管理
│   │   │   ├── prediction/        # 预测分析
│   │   │   ├── diagnosis/         # 故障诊断
│   │   │   └── settings/          # 系统设置
│   │   ├── stores/                # 状态管理(Zustand)
│   │   ├── services/              # API服务
│   │   ├── hooks/                 # 自定义Hooks
│   │   ├── utils/                 # 工具函数
│   │   ├── types/                 # TypeScript类型
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── backend/                       # 后端项目 (FastAPI)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI应用入口
│   │   ├── core/                  # 核心配置
│   │   ├── api/                   # API路由
│   │   ├── models/                # 数据模型
│   │   ├── services/              # 业务服务
│   │   └── utils/                 # 工具函数
│   ├── tests/                     # 测试代码
│   ├── alembic/                   # 数据库迁移
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml/                            # 机器学习模块
│   ├── data/                      # 数据处理
│   ├── models/                    # 模型定义
│   ├── training/                  # 训练脚本
│   ├── inference/                 # 推理服务
│   └── evaluation/                # 模型评估
│
├── kg/                            # 知识图谱模块
│   ├── ontology/                  # 本体定义
│   ├── construction/              # 图谱构建
│   ├── retrieval/                 # 图谱检索
│   └── rag/                       # GraphRAG引擎
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后数据
│   ├── features/                  # 特征数据
│   ├── models/                    # 模型权重
│   └── knowledge/                 # 知识库数据
│
├── scripts/                       # 脚本工具
├── docker/                        # Docker配置
├── tests/                         # 集成测试
├── .env.example                   # 环境变量示例
├── .gitignore
├── Makefile                       # 常用命令
└── README.md                      # 项目说明
```

### 2.2 模块职责说明

| 模块 | 职责 | 主要技术 |
|-----|------|---------|
| frontend | 用户界面、数据可视化、交互逻辑 | React, TypeScript, Ant Design, ECharts |
| backend | API服务、业务逻辑、数据管理 | FastAPI, SQLAlchemy, Pydantic |
| ml | 模型训练、推理、评估 | PyTorch, NumPy, Scikit-learn |
| kg | 知识图谱构建、检索、RAG推理 | Neo4j, LangChain |

---

## 3. 开发规范

### 3.1 代码规范

#### 3.1.1 Python代码规范 (PEP 8)

```python
"""
模块文档字符串示例。
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn


class XLSTMBlock(nn.Module):
    """
    xLSTM块实现。
    
    结合mLSTM（矩阵记忆）和sLSTM（标量门控）的混合架构。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ) -> None:
        """初始化xLSTM块。"""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.mlstm = MatrixLSTM(input_dim, hidden_dim, num_layers)
        self.slstm = ScalarLSTM(input_dim, hidden_dim, num_layers)
        self.gate = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {x.size(-1)}"
            )
        
        h_m = self.mlstm(x)
        h_s = self.slstm(x)
        
        gate_weights = torch.softmax(
            self.gate(torch.cat([h_m, h_s], dim=-1)),
            dim=-1
        )
        
        output = gate_weights[..., 0:1] * h_m + gate_weights[..., 1:2] * h_s
        
        return self.dropout(output)
```

**命名规范**：

| 类型 | 规范 | 示例 |
|-----|------|------|
| 模块名 | 小写下划线 | `data_preprocessing.py` |
| 类名 | 大驼峰 | `BatteryPredictor` |
| 函数名 | 小写下划线 | `predict_rul()` |
| 变量名 | 小写下划线 | `battery_capacity` |
| 常量名 | 大写下划线 | `MAX_CYCLES` |

#### 3.1.2 TypeScript代码规范

```typescript
/**
 * 电池数据接口
 */
export interface Battery {
  id: number;
  name: string;
  type: string;
  ratedCapacity: number;
  currentSoh?: number;
  createdAt: string;
}

/**
 * 电池服务类
 */
class BatteryService {
  private readonly baseUrl: string;
  
  constructor(baseUrl: string = '/api/v1') {
    this.baseUrl = baseUrl;
  }
  
  async getBatteries(
    params: PaginationParams = {}
  ): Promise<PaginatedResponse<Battery>> {
    const response = await api.get(`${this.baseUrl}/data/batteries`, {
      params: {
        page: params.page ?? 1,
        pageSize: params.pageSize ?? 20,
        ...params
      }
    });
    
    return response.data;
  }
}

export const batteryService = new BatteryService();
```

### 3.2 Git工作流

#### 3.2.1 分支策略 (Git Flow)

```
main (生产分支)
  │
  ├── develop (开发分支)
  │     │
  │     ├── feature/xlstm-implementation
  │     ├── feature/graph-rag
  │     │
  │     └── release/v1.0.0 (发布分支)
  │
  └── hotfix/security-patch (热修复分支)
```

#### 3.2.2 提交规范 (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型说明**:
| 类型 | 说明 | 示例 |
|-----|------|------|
| feat | 新功能 | `feat(prediction): add batch prediction API` |
| fix | 修复bug | `fix(database): fix connection pool leak` |
| docs | 文档更新 | `docs(api): update authentication docs` |
| refactor | 重构 | `refactor(kg): simplify graph query logic` |
| test | 测试 | `test(backend): add unit tests for auth` |

### 3.3 代码审查规范

**审查清单**:

- [ ] 功能实现符合需求文档
- [ ] 边界条件处理正确
- [ ] 错误处理完善
- [ ] 代码符合规范 (PEP 8 / ESLint)
- [ ] 命名清晰有意义
- [ ] 单元测试覆盖核心逻辑
- [ ] 无硬编码敏感信息

---

## 4. 模块实现指南

### 4.1 数据管理模块

#### 4.1.1 数据获取

**数据源**：

| 数据集 | 来源 | 说明 |
|-------|------|------|
| NASA PCoE | NASA官网 | 锂电池老化数据 |
| CALCE | CALCE官网 | 多种电池型号的老化数据 |
| Kaggle | Kaggle平台 | 社区贡献的电池数据集 |

**下载脚本**：

```python
import os
import requests
from pathlib import Path


def download_nasa_dataset(save_dir: str = "data/raw"):
    """下载NASA电池老化数据集。"""
    urls = {
        "B0005": "https://ti.arc.nasa.gov/c/27/B0005.mat",
        "B0006": "https://ti.arc.nasa.gov/c/27/B0006.mat",
        "B0007": "https://ti.arc.nasa.gov/c/27/B0007.mat",
        "B0018": "https://ti.arc.nasa.gov/c/27/B0018.mat",
    }

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for name, url in urls.items():
        save_path = os.path.join(save_dir, f"{name}.mat")
        if not os.path.exists(save_path):
            print(f"Downloading {name}...")
            response = requests.get(url)
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {name}")
```

#### 4.1.2 数据预处理

```python
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler


class BatteryDataPreprocessor:
    """电池数据预处理器。"""

    def __init__(
        self,
        filter_window: int = 5,
        interpolation_method: str = "cubic",
    ):
        self.filter_window = filter_window
        self.interpolation_method = interpolation_method
        self.scaler = StandardScaler()

    def load_mat_data(self, file_path: str) -> pd.DataFrame:
        """加载MAT格式数据文件。"""
        from scipy.io import loadmat

        data = loadmat(file_path)
        battery_data = data[list(data.keys())[3]]

        cycles = []
        capacities = []
        for i, cycle in enumerate(battery_data[0][0][0][0]):
            if cycle[0][0][0].size > 0:
                capacity = cycle[0][0][0][0][0]
                cycles.append(i + 1)
                capacities.append(capacity)

        return pd.DataFrame({"cycle": cycles, "capacity": capacities})

    def smooth_data(
        self, data: np.ndarray, method: str = "savgol"
    ) -> np.ndarray:
        """数据平滑处理。"""
        if method == "savgol":
            return signal.savgol_filter(
                data, self.filter_window, polyorder=2
            )
        elif method == "moving_average":
            kernel = np.ones(self.filter_window) / self.filter_window
            return np.convolve(data, kernel, mode="same")
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def extract_health_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """提取健康因子。"""
        df = df.copy()

        df["capacity_fade_rate"] = (
            df["capacity"].iloc[0] - df["capacity"]
        ) / df["capacity"].iloc[0]

        df["soh"] = df["capacity"] / df["capacity"].iloc[0] * 100

        df["capacity_smoothed"] = self.smooth_data(
            df["capacity"].values
        )

        df["delta_capacity"] = np.gradient(df["capacity_smoothed"])

        return df

    def process_pipeline(
        self, file_path: str, save_path: str = None
    ) -> pd.DataFrame:
        """完整预处理流程。"""
        df = self.load_mat_data(file_path)
        df = self.extract_health_indicators(df)

        if save_path:
            df.to_csv(save_path, index=False)

        return df
```

### 4.2 xLSTM-Transformer混合模型开发

#### 4.2.1 xLSTM模块实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class mLSTMCell(nn.Module):
    """矩阵记忆LSTM单元。"""

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(
            torch.randn(4 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.register_parameter("bias", None)

        self.matrix_memory = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
        )

    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_prev, c_prev, m_prev = state

        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(
            h_prev, self.weight_hh
        )

        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        m_new = f.unsqueeze(-1) * m_prev + i.unsqueeze(-1) * g.unsqueeze(-2)
        c_new = f * c_prev + i * g

        h_new = o * torch.tanh(
            torch.matmul(c_new, self.matrix_memory) + c_new
        )

        return h_new, (h_new, c_new, m_new)


class xLSTMBlock(nn.Module):
    """xLSTM块，包含mLSTM和sLSTM。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.mlstm_layers = nn.ModuleList(
            [
                mLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(
        self, x: torch.Tensor, states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        batch_size, seq_len, _ = x.shape

        if states is None:
            states = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(
                        batch_size,
                        self.hidden_size,
                        self.hidden_size,
                        device=x.device,
                    ),
                )
                for _ in range(self.num_layers)
            ]

        outputs = []
        new_states = []

        for t in range(seq_len):
            h = x[:, t, :]
            layer_states = []

            for layer_idx, (mlstm, ln) in enumerate(
                zip(self.mlstm_layers, self.layer_norms)
            ):
                h, state = mlstm(h, states[layer_idx])
                h = ln(h)
                h = self.dropout(h)
                layer_states.append(state)

            outputs.append(h)
            new_states.append(layer_states)

        output = torch.stack(outputs, dim=1)
        output = self.proj(output)

        return output, new_states[-1]
```

#### 4.2.2 Transformer模块实现

```python
class TransformerBlock(nn.Module):
    """Transformer块。"""

    def __init__(
        self,
        d_model: int,
        n_head: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
```

#### 4.2.3 混合模型实现

```python
class HybridModel(nn.Module):
    """xLSTM-Transformer混合模型。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        n_head: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        self.xlstm = xLSTMBlock(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_head, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_gate = nn.Linear(hidden_dim * 2, 2)
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        
        xlstm_out, _ = self.xlstm(x)
        
        trans_out = x
        for block in self.transformer_blocks:
            trans_out = block(trans_out)
        
        concat = torch.cat([xlstm_out, trans_out], dim=-1)
        gate = torch.softmax(self.fusion_gate(concat), dim=-1)
        
        fused = gate[..., 0:1] * xlstm_out + gate[..., 1:2] * trans_out
        fused = self.fusion(concat)
        
        output = self.output_head(fused)
        
        return output.squeeze(-1)
```

---

## 5. 模型训练完整流程

### 5.1 数据准备

```python
# ml/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class BatteryDataset(Dataset):
    """电池数据集。"""
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 50,
        prediction_horizon: int = 1,
        transform: Optional[callable] = None
    ):
        self.data = pd.read_csv(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform
        
        self.features = self._extract_features()
        self.labels = self._extract_labels()
        
    def _extract_features(self) -> np.ndarray:
        """提取特征。"""
        feature_cols = ['capacity', 'soh', 'capacity_fade_rate']
        return self.data[feature_cols].values
    
    def _extract_labels(self) -> np.ndarray:
        """提取标签（RUL）。"""
        max_cycle = self.data['cycle'].max()
        return max_cycle - self.data['cycle'].values
    
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.prediction_horizon
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx:idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length]
        
        if self.transform:
            features = self.transform(features)
        
        return torch.FloatTensor(features), torch.FloatTensor([label])


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    sequence_length: int = 50,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器。"""
    dataset = BatteryDataset(data_path, sequence_length)
    
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
```

### 5.2 训练器实现

```python
# ml/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
from tqdm import tqdm
import os


class Trainer:
    """模型训练器。"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = nn.HuberLoss(delta=1.0)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        self.writer = SummaryWriter(config.get('log_dir', 'runs'))
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 20)
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch。"""
        self.model.train()
        total_loss = 0
        total_metrics = {'mae': 0, 'rmse': 0}
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (features, labels) in enumerate(progress_bar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            mae = torch.mean(torch.abs(outputs - labels)).item()
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
            total_metrics['mae'] += mae
            total_metrics['rmse'] += rmse
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mae': mae,
                'rmse': rmse
            })
        
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'mae': total_metrics['mae'] / num_batches,
            'rmse': total_metrics['rmse'] / num_batches
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """验证。"""
        self.model.eval()
        total_loss = 0
        total_metrics = {'mae': 0, 'rmse': 0}
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                total_metrics['mae'] += torch.mean(torch.abs(outputs - labels)).item()
                total_metrics['rmse'] += torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
        
        num_batches = len(self.val_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'mae': total_metrics['mae'] / num_batches,
            'rmse': total_metrics['rmse'] / num_batches
        }
        
        return metrics
    
    def train(self, num_epochs: int) -> None:
        """完整训练流程。"""
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f'val/{name}', value, epoch)
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MAE: {train_metrics['mae']:.4f}, "
                  f"RMSE: {train_metrics['rmse']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"MAE: {val_metrics['mae']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n早停：验证损失{self.patience}轮未改善")
                    break
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_metrics)
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict) -> None:
        """保存检查点。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        save_path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save(checkpoint, save_path)
        print(f"检查点已保存: {save_path}")
```

---

## 6. 错误处理策略

### 6.1 自定义异常类

```python
# backend/app/core/exceptions.py
from fastapi import HTTPException, status


class BHMSException(Exception):
    """BHMS基础异常类。"""
    
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ResourceNotFoundError(BHMSException):
    """资源未找到异常。"""
    
    def __init__(self, resource_type: str, resource_id: int):
        message = f"{resource_type} with id {resource_id} not found"
        super().__init__(message, code="RESOURCE_NOT_FOUND")


class ModelInferenceError(BHMSException):
    """模型推理异常。"""
    
    def __init__(self, detail: str):
        message = f"Model inference failed: {detail}"
        super().__init__(message, code="MODEL_INFERENCE_ERROR")


class DataValidationError(BHMSException):
    """数据验证异常。"""
    
    def __init__(self, field: str, reason: str):
        message = f"Validation failed for field '{field}': {reason}"
        super().__init__(message, code="DATA_VALIDATION_ERROR")


class KGQueryError(BHMSException):
    """知识图谱查询异常。"""
    
    def __init__(self, query: str, detail: str):
        message = f"Knowledge graph query failed: {detail}"
        super().__init__(message, code="KG_QUERY_ERROR")
```

### 6.2 全局异常处理器

```python
# backend/app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog

from app.core.exceptions import BHMSException

logger = structlog.get_logger()

app = FastAPI()


@app.exception_handler(BHMSException)
async def bhms_exception_handler(request: Request, exc: BHMSException):
    """BHMS异常处理器。"""
    logger.error(
        "bhms_exception",
        code=exc.code,
        message=exc.message,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "code": exc.code,
            "message": exc.message,
            "path": request.url.path
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器。"""
    logger.error(
        "validation_error",
        errors=exc.errors(),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器。"""
    logger.exception(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "path": request.url.path
        }
    )
```

---

## 7. 前端状态管理

### 7.1 Zustand状态管理

```typescript
// frontend/src/stores/userStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authService } from '@/services/authService';
import type { User, LoginCredentials } from '@/types';

interface UserState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  refreshAccessToken: () => Promise<boolean>;
  updateUser: (userData: Partial<User>) => void;
  clearError: () => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      
      login: async (credentials) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await authService.login(credentials);
          
          set({
            user: response.user,
            token: response.access_token,
            refreshToken: response.refresh_token,
            isAuthenticated: true,
            isLoading: false
          });
        } catch (error: any) {
          set({
            error: error.message || '登录失败',
            isLoading: false,
            isAuthenticated: false
          });
          throw error;
        }
      },
      
      logout: () => {
        authService.logout();
        
        set({
          user: null,
          token: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null
        });
      },
      
      refreshAccessToken: async () => {
        const { refreshToken } = get();
        
        if (!refreshToken) {
          return false;
        }
        
        try {
          const response = await authService.refreshToken(refreshToken);
          
          set({
            token: response.access_token,
            isAuthenticated: true
          });
          
          return true;
        } catch (error) {
          get().logout();
          return false;
        }
      },
      
      updateUser: (userData) => {
        const { user } = get();
        
        if (user) {
          set({
            user: { ...user, ...userData }
          });
        }
      },
      
      clearError: () => {
        set({ error: null });
      }
    }),
    {
      name: 'user-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated
      })
    }
  )
);
```

---

## 8. API开发指南

### 8.1 API开发规范

```python
# backend/app/api/v1/prediction.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models.schemas import (
    PredictionRequest,
    PredictionResult,
    PredictionHistory,
    BatchPredictionRequest
)
from app.services.prediction_service import PredictionService
from app.models.database import User

router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.post(
    "/rul",
    response_model=PredictionResult,
    summary="执行RUL预测",
    description="基于电池历史数据预测剩余使用寿命",
    responses={
        200: {"description": "预测成功"},
        400: {"description": "参数错误"},
        404: {"description": "电池不存在"},
        500: {"description": "模型推理失败"}
    }
)
async def predict_rul(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    执行RUL预测。
    
    - **battery_id**: 电池ID (必填)
    - **cycle_number**: 起始循环数 (可选，默认为最新)
    - **model_version**: 模型版本 (可选，默认使用最新)
    - **return_curve**: 是否返回预测曲线 (可选，默认true)
    """
    service = PredictionService(db)
    
    try:
        result = await service.predict_rul(request)
        return result
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelInferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch",
    response_model=List[PredictionResult],
    summary="批量RUL预测",
    description="对多个电池执行批量预测"
)
async def batch_predict_rul(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """批量预测RUL"""
    if len(request.battery_ids) > 100:
        raise HTTPException(
            status_code=400,
            detail="单次批量预测最多支持100个电池"
        )
    
    service = PredictionService(db)
    results = []
    
    for battery_id in request.battery_ids:
        try:
            result = await service.predict_rul(
                PredictionRequest(
                    battery_id=battery_id,
                    model_version=request.model_version
                )
            )
            results.append(result)
        except Exception as e:
            logger.error(f"预测电池 {battery_id} 失败: {e}")
    
    return results


@router.get(
    "/history/{battery_id}",
    response_model=PredictionHistory,
    summary="获取预测历史",
    description="获取指定电池的预测历史记录"
)
async def get_prediction_history(
    battery_id: int,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    model_version: Optional[str] = Query(None, description="模型版本过滤"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取预测历史"""
    service = PredictionService(db)
    
    history = await service.get_history(
        battery_id=battery_id,
        page=page,
        page_size=page_size,
        model_version=model_version
    )
    
    return history
```

---

## 9. 测试策略

### 9.1 测试金字塔

```
                    ┌─────────┐
                    │   E2E   │  5%  (端到端测试)
                    │  Tests  │
                   ┌┴─────────┴┐
                   │ Integration│ 15%  (集成测试)
                   │   Tests   │
                  ┌┴───────────┴┐
                  │    Unit     │ 80%  (单元测试)
                  │    Tests    │
                  └─────────────┘
```

### 9.2 单元测试示例

```python
# backend/tests/unit/test_prediction_service.py
import pytest
from unittest.mock import Mock, patch

from app.services.prediction_service import PredictionService
from app.models.schemas import PredictionRequest


class TestPredictionService:
    """预测服务单元测试"""
    
    @pytest.fixture
    def mock_db(self):
        """模拟数据库会话"""
        return Mock()
    
    @pytest.fixture
    def service(self, mock_db):
        """预测服务实例"""
        with patch('app.services.prediction_service.RULPredictor'):
            service = PredictionService(mock_db)
            return service
    
    async def test_predict_rul_success(self, service, mock_db):
        """测试预测成功场景"""
        request = PredictionRequest(battery_id=1)
        
        mock_battery = Mock()
        mock_battery.id = 1
        mock_battery.current_cycle = 500
        mock_battery.current_soh = 85.5
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_battery
        
        service.predictor.predict.return_value = {
            'rul': 350,
            'confidence': 0.92
        }
        
        result = await service.predict_rul(request)
        
        assert result.battery_id == 1
        assert result.predicted_rul == 350
        assert result.confidence == 0.92
    
    async def test_predict_rul_battery_not_found(self, service, mock_db):
        """测试电池不存在场景"""
        request = PredictionRequest(battery_id=999)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ResourceNotFoundError) as exc_info:
            await service.predict_rul(request)
        
        assert "not found" in str(exc_info.value)
```

### 9.3 测试命令

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit -v

# 运行集成测试
pytest tests/integration -v

# 生成覆盖率报告
pytest --cov=app --cov-report=html

# 运行特定测试
pytest tests/unit/test_prediction_service.py::TestPredictionService::test_predict_rul_success -v
```

---

## 10. CI/CD流程

### 10.1 CI/CD架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │────▶│   GitHub    │────▶│   Docker    │────▶│   Deploy    │
│    Push     │     │   Actions   │     │    Hub      │     │  to Server  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Tests     │
                    │  - Lint     │
                    │  - Unit     │
                    │  - Security │
                    └─────────────┘
```

### 10.2 GitHub Actions配置

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install black flake8 isort mypy
          pip install -r requirements.txt
      
      - name: Run Black
        run: black --check .
      
      - name: Run Flake8
        run: flake8 .
      
      - name: Run isort
        run: isort --check-only .
      
      - name: Run mypy
        run: mypy app/

  test:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5.15-community
        env:
          NEO4J_AUTH: neo4j/testpassword
        ports:
          - 7687:7687
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest tests/unit -v --cov=app --cov-report=xml
        env:
          DATABASE_URL: sqlite:///./test.db
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: testpassword
          REDIS_URL: redis://localhost:6379/0
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push backend
        uses: docker/build-push-action@v4
        with:
          context: ./backend
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/bhms-backend:latest
            ${{ secrets.DOCKER_USERNAME }}/bhms-backend:${{ github.sha }}
      
      - name: Build and push frontend
        uses: docker/build-push-action@v4
        with:
          context: ./frontend
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/bhms-frontend:latest
            ${{ secrets.DOCKER_USERNAME }}/bhms-frontend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            cd /opt/bhms
            docker-compose pull
            docker-compose up -d
            docker system prune -f
```

---

## 11. 部署指南

### 11.1 生产环境部署

#### 11.1.1 服务器准备

```bash
# 1. 系统更新
sudo apt update && sudo apt upgrade -y

# 2. 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. 创建项目目录
mkdir -p /opt/bhms
cd /opt/bhms
```

#### 11.1.2 部署步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-org/bhms.git .

# 2. 创建环境变量文件
cat > .env << EOF
DATABASE_URL=sqlite:///data/bhms.db
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password_here
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-super-secret-key-min-32-chars-long
LLM_API_KEY=your-llm-api-key
EOF

# 3. 创建数据目录
mkdir -p data models logs

# 4. 启动服务
docker-compose up -d

# 5. 检查服务状态
docker-compose ps

# 6. 初始化数据库
docker-compose exec backend alembic upgrade head

# 7. 初始化知识图谱
docker-compose exec backend python scripts/init_kg.py
```

### 11.2 部署验证

```bash
# 检查API健康状态
curl http://localhost:8000/api/v1/health

# 检查前端访问
curl -I http://localhost:80

# 查看日志
docker-compose logs -f backend
docker-compose logs -f frontend
```

---

## 12. 监控与运维

### 12.1 监控配置

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bhms-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 12.2 告警规则

```yaml
# monitoring/alert_rules.yml
groups:
  - name: bhms_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: PredictionLatencyHigh
        expr: histogram_quantile(0.99, rate(prediction_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Prediction latency is high"
```

### 12.3 运维命令

```bash
# 查看服务状态
docker-compose ps

# 查看资源使用
docker stats

# 查看日志
docker-compose logs -f --tail=100

# 备份数据库
docker-compose exec backend sqlite3 /app/data/bhms.db ".backup '/backup/bhms_$(date +%Y%m%d).db'"

# 重启服务
docker-compose restart backend

# 更新部署
docker-compose pull && docker-compose up -d
```

---

## 13. 附录

### 13.1 常用命令速查

| 命令 | 说明 |
|-----|------|
| `make dev` | 启动开发环境 |
| `make test` | 运行测试 |
| `make lint` | 代码检查 |
| `make build` | 构建镜像 |
| `make deploy` | 部署到生产 |

### 13.2 故障排查

| 问题 | 排查步骤 | 解决方案 |
|-----|---------|---------|
| 服务启动失败 | 查看日志 `docker-compose logs` | 检查配置、依赖 |
| 数据库连接失败 | 检查网络、凭据 | 验证.env配置 |
| 模型推理失败 | 检查GPU、模型文件 | 重新加载模型 |
| 内存不足 | 查看`docker stats` | 增加内存限制 |

### 13.3 参考资料

1. FastAPI官方文档: https://fastapi.tiangolo.com/
2. PyTorch官方文档: https://pytorch.org/docs/
3. Neo4j官方文档: https://neo4j.com/docs/
4. React官方文档: https://react.dev/
5. Zustand文档: https://docs.pmnd.rs/zustand/

---

**文档结束**
