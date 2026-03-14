# 锂电池健康管理系统（BHMS）设计文档

## 文档控制信息

| 属性 | 内容 |
|-----|------|
| 项目名称 | 锂电池健康管理系统（Battery Health Management System） |
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
| V1.0 | 2026-03-10 | BHMS团队 | 初稿创建 |
| V2.0 | 2026-03-10 | BHMS团队 | 整合V1和V2版本，补充完整架构设计 |

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构设计](#2-系统架构设计)
3. [模块划分设计](#3-模块划分设计)
4. [核心算法设计](#4-核心算法设计)
5. [数据库设计](#5-数据库设计)
6. [接口设计](#6-接口设计)
7. [安全设计](#7-安全设计)
8. [性能设计](#8-性能设计)

---

## 1. 项目概述

### 1.1 项目背景

随着"双碳"战略的实施，锂离子电池的健康管理（BMS）已成为行业痛点。电池老化是一个跨越数千次循环的复杂过程。当前主流BMS软件系统在核心算法层面面临两大瓶颈：

1. **算力层面**：Transformer在处理全生命周期（>3000 Cycles）数据时显存呈指数级增长，难以在工业级系统中落地
2. **预测和解释层面**：现有深度学习模型大多为黑盒，缺乏可解释性，非专业用户无法理解系统提示

### 1.2 项目目标

构建一个集寿命预测、故障归因与实时监控于一体的锂电池健康管理系统，实现：

1. 基于xLSTM与Transformer混合架构的精准RUL预测
2. 基于GraphRAG的可解释性故障诊断
3. 友好的Web可视化交互界面
4. 工业级实时响应能力

### 1.3 性能指标

| 指标类别 | 具体指标 | 目标值 |
|---------|---------|--------|
| 预测精度 | RMSE | 显著优于传统LSTM模型（降低20%以上） |
| 推理效率 | 推理速度 | 较纯Transformer架构提升30%以上 |
| 序列处理 | 最大循环数 | 支持>3000 Cycles全生命周期数据 |
| 显存占用 | 峰值显存 | 线性复杂度，显著降低显存需求 |
| 响应时间 | API响应 | <500ms（单次预测请求） |

---

## 2. 系统架构设计

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户层（Presentation Layer）                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   数据上传模块    │  │   可视化展示     │  │   诊断报告查看   │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              服务层（Service Layer）                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        FastAPI 后端服务                               │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │  数据处理API  │  │  预测服务API  │  │  诊断服务API  │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              核心算法层（Algorithm Layer）                    │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐          │
│  │    xLSTM-Transformer混合模型  │  │      GraphRAG诊断引擎       │          │
│  │  ┌─────────┐  ┌─────────┐   │  │  ┌─────────┐  ┌─────────┐   │          │
│  │  │ xLSTM   │  │Transformer│  │  │  │ Neo4j  │  │  LLM    │   │          │
│  │  │ 模块    │  │  模块    │  │  │  │ 知识图谱│  │ 推理引擎│   │          │
│  │  └─────────┘  └─────────┘   │  │  └─────────┘  └─────────┘   │          │
│  │        特征融合层            │  │      子图检索 + 语义映射     │          │
│  └─────────────────────────────┘  └─────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据层（Data Layer）                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  原始数据存储  │  │  特征数据库   │  │  知识图谱存储  │  │  模型权重存储  │    │
│  │   (CSV/HDF5) │  │   (SQLite)   │  │   (Neo4j)    │  │  (.pt/.bin)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈选型

| 层级 | 技术选型 | 选型理由 |
|-----|---------|---------|
| 前端框架 | React 18+ | 组件化开发、生态丰富、性能优秀 |
| 可视化库 | ECharts 5+ | 支持大数据量渲染、图表类型丰富 |
| 后端框架 | FastAPI | 高性能异步框架、自动API文档生成 |
| 深度学习框架 | PyTorch 2.0+ | 动态图机制、便于模型调试与创新 |
| 知识图谱数据库 | Neo4j | 成熟的图数据库、支持Cypher查询 |
| 大模型框架 | LangChain | 模型集成便捷、支持RAG架构 |
| 向量数据库 | ChromaDB | 轻量级、支持本地部署 |

---

## 3. 模块划分设计

### 3.1 模块总览

```
BHMS/
├── frontend/                    # 前端模块
│   ├── src/
│   │   ├── components/          # UI组件
│   │   ├── pages/               # 页面
│   │   ├── services/            # API服务
│   │   ├── hooks/               # 自定义Hooks
│   │   └── utils/               # 工具函数
│   └── package.json
│
├── backend/                     # 后端模块
│   ├── api/                     # API路由
│   ├── core/                    # 核心配置
│   ├── models/                  # 数据模型
│   ├── services/                # 业务服务
│   └── utils/                   # 工具函数
│
├── ml/                          # 机器学习模块
│   ├── data/                    # 数据处理
│   │   ├── preprocessing/       # 数据预处理
│   │   └── feature_engineering/ # 特征工程
│   ├── models/                  # 模型定义
│   │   ├── xlstm/               # xLSTM模块
│   │   ├── transformer/         # Transformer模块
│   │   └── hybrid/              # 混合模型
│   ├── training/                # 训练脚本
│   └── inference/               # 推理服务
│
├── kg/                          # 知识图谱模块
│   ├── ontology/                # 本体定义
│   ├── construction/            # 图谱构建
│   ├── retrieval/               # 图谱检索
│   └── rag/                     # GraphRAG引擎
│
└── data/                        # 数据目录
    ├── raw/                     # 原始数据
    ├── processed/               # 处理后数据
    ├── features/                # 特征数据
    └── models/                  # 模型权重
```

### 3.2 核心模块详细设计

#### 3.2.1 数据预处理模块

**功能描述**：对原始电池监测数据进行清洗、标准化和特征提取

**输入**：原始电池数据（电压、电流、温度、容量等时序数据）

**输出**：标准化特征序列、健康因子（HIs）

**处理流程**：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  数据加载    │───▶│  异常值处理  │───▶│  滤波平滑    │───▶│  特征提取    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                        ┌─────────────┐
                                                        │  标准化输出  │
                                                        └─────────────┘
```

**关键特征**：

| 特征名称 | 计算方法 | 物理意义 |
|---------|---------|---------|
| IC曲线峰值 | dQ/dV计算后取峰值 | 反映电池内部化学反应特性 |
| 恒流充电时间 | CC阶段持续时间 | 反映电池充电接受能力 |
| 放电电压差 | 放电末端电压-起始电压 | 反映电池放电平台特性 |
| 容量衰减率 | (C_n - C_0) / C_0 | 反映电池老化程度 |
| 温度变化率 | dT/dt | 反映电池热特性 |

#### 3.2.2 xLSTM-Transformer混合模型模块

**架构设计**：

```
输入序列 (B, L, D)
        │
        ▼
┌───────────────────┐
│   输入嵌入层       │
│   (Linear + PE)   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   xLSTM Block 1   │  ◀── mLSTM: 矩阵记忆，线性复杂度
│   (mLSTM + sLSTM) │      sLSTM: 指数门控，增强非线性
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Transformer Block │  ◀── 多头自注意力，捕捉局部特征
│   (MHSA + FFN)    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   xLSTM Block 2   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Transformer Block │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   特征融合层       │  ◀── 加权融合 + 残差连接
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   输出预测头       │
│   (MLP + Sigmoid) │
└───────────────────┘
        │
        ▼
RUL预测值 (B, 1)
```

**模块职责划分**：

| 模块 | 职责 | 复杂度 |
|-----|------|-------|
| xLSTM | 处理超长序列，捕捉长期退化趋势 | O(L) |
| Transformer | 捕捉局部瞬时波动特征 | O(L²) |
| 特征融合层 | 整合两种特征表示 | O(D) |

#### 3.2.3 GraphRAG诊断引擎模块

**整体架构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GraphRAG故障诊断引擎架构                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        输入层 (Input Layer)                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  电池实时数据 / 历史数据                                      │   │   │
│  │  │  • 电压曲线  • 电流曲线  • 温度曲线  • 容量数据              │   │   │
│  │  │  • 内阻数据  • 循环次数  • 充放电时间                        │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    异常检测层 (Anomaly Detection)                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │   统计阈值法     │  │   孤立森林算法   │  │   深度异常检测   │     │   │
│  │  │  (Rule-based)   │  │  (Isolation     │  │  (Autoencoder)  │     │   │
│  │  │                 │  │   Forest)       │  │                 │     │   │
│  │  │ • 容量衰减阈值   │  │                 │  │                 │     │   │
│  │  │   >20%: 异常    │  │ • 无监督学习    │  │ • 重构误差检测  │     │   │
│  │  │ • 电压跳变阈值   │  │ • 适合多维特征  │  │ • 适合复杂模式  │     │   │
│  │  │   >0.1V: 异常   │  │ • 异常分数输出  │  │ • 需要训练数据  │     │   │
│  │  │ • 温度异常阈值   │  │                 │  │                 │     │   │
│  │  │   >45°C: 异常   │  │                 │  │                 │     │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │   │
│  │                                                                     │   │
│  │  输出: 异常特征列表 ["容量骤降", "电压平台异常", "温度异常升高等"]      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  知识图谱检索层 (Knowledge Retrieval)                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Step 1: 实体识别与链接                                      │   │   │
│  │  │  • 将异常特征映射到知识图谱中的Symptom实体                   │   │   │
│  │  │  • 示例: "容量骤降" → (:Symptom {name: "容量骤降"})          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Step 2: 子图检索 (Cypher查询)                               │   │   │
│  │  │                                                              │   │   │
│  │  │  MATCH (s:Symptom) WHERE s.name IN $symptoms                 │   │   │
│  │  │  MATCH (f:Fault)-[:HAS_SYMPTOM]->(s)                         │   │   │
│  │  │  OPTIONAL MATCH (f)-[:CAUSED_BY]->(c:Cause)                  │   │   │
│  │  │  OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(sol:Solution)          │   │   │
│  │  │  RETURN f, collect(DISTINCT c) as causes,                    │   │   │
│  │  │         collect(DISTINCT sol) as solutions                   │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Step 3: 上下文构建                                          │   │   │
│  │  │  • 整合故障类型、原因、解决方案信息                          │   │   │
│  │  │  • 构建结构化上下文供LLM使用                                 │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  输出: 结构化知识上下文 (JSON格式)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LLM推理层 (LLM Reasoning)                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Prompt模板 (可配置)                                         │   │   │
│  │  │                                                              │   │   │
│  │  │  你是一位专业的锂电池故障诊断专家。请根据以下信息生成诊断报告。  │   │   │
│  │  │                                                              │   │   │
│  │  │  【检测到的异常特征】                                         │   │   │
│  │  │  {{symptoms}}                                                │   │   │
│  │  │                                                              │   │   │
│  │  │  【知识图谱检索结果】                                         │   │   │
│  │  │  {{knowledge_context}}                                       │   │   │
│  │  │                                                              │   │   │
│  │  │  请生成一份详细的诊断报告，包括：                              │   │   │
│  │  │  1. 故障类型判断及置信度                                       │   │   │
│  │  │  2. 故障原因分析（基于物理机理）                               │   │   │
│  │  │  3. 严重程度评估                                              │   │   │
│  │  │  4. 建议措施（按优先级排序）                                   │   │   │
│  │  │  5. 预防措施建议                                              │   │   │
│  │  │                                                              │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  LLM模型选项:                                                       │   │
│  │  • 云端API: DeepSeek-V3, GPT-4, Claude-3                          │   │
│  │  • 本地部署: DeepSeek-R1-Distill, Qwen-7B-Chat                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      输出层 (Output Layer)                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  诊断报告 (结构化JSON + 自然语言文本)                         │   │   │
│  │  │  {                                                          │   │   │
│  │  │    "diagnosis_id": "uuid",                                  │   │   │
│  │  │    "battery_id": 123,                                       │   │   │
│  │  │    "timestamp": "2026-03-10T10:00:00Z",                     │   │   │
│  │  │    "detected_symptoms": ["容量骤降", "电压平台异常"],        │   │   │
│  │  │    "fault_type": {                                          │   │   │
│  │  │      "name": "析锂",                                        │   │   │
│  │  │      "confidence": 0.85,                                    │   │   │
│  │  │      "severity": "high"                                     │   │   │
│  │  │    },                                                       │   │   │
│  │  │    "causes": ["过充", "低温充电"],                           │   │   │
│  │  │    "solutions": [                                           │   │   │
│  │  │      {"action": "降低充电截止电压", "priority": 1},          │   │   │
│  │  │      {"action": "提高环境温度", "priority": 2}               │   │   │
│  │  │    ],                                                       │   │   │
│  │  │    "report_text": "根据分析，电池当前存在...",               │   │   │
│  │  │    "recommended_action": "建议立即调整充电策略..."           │   │   │
│  │  │  }                                                          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**异常检测算法详细设计**：

| 算法 | 适用场景 | 阈值/参数 | 输出 |
|-----|---------|----------|------|
| 统计阈值法 | 单变量异常检测 | 容量衰减率>20%, 电压跳变>0.1V, 温度>45°C | 布尔值 |
| 孤立森林 | 多变量异常检测 | n_estimators=100, contamination=0.1 | 异常分数(0-1) |
| 深度自编码器 | 复杂模式检测 | hidden_dims=[64, 32, 64], threshold=μ+2σ | 重构误差 |

**知识图谱本体设计**：

```
                    ┌─────────────┐
                    │   Battery   │
                    └─────────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │ Symptom │   │  Fault  │   │  Cause  │
      └─────────┘   └─────────┘   └─────────┘
            │             │             │
            │             │             │
            └─────────────┼─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Solution   │
                    └─────────────┘
```

**实体类型定义**：

| 实体类型 | 属性 | 示例 |
|---------|------|------|
| Battery | id, type, capacity, cycles | 18650锂电池 |
| Symptom | name, threshold, description | 容量骤降 |
| Fault | name, severity, mechanism | 析锂故障 |
| Cause | name, probability, condition | 过充 |
| Solution | action, priority, reference | 降低充电倍率 |

**关系类型定义**：

| 关系类型 | 起始实体 | 目标实体 | 含义 |
|---------|---------|---------|------|
| HAS_SYMPTOM | Battery | Symptom | 电池表现症状 |
| INDICATES | Symptom | Fault | 症状指示故障 |
| CAUSED_BY | Fault | Cause | 故障原因 |
| HAS_SOLUTION | Fault | Solution | 故障解决方案 |

---

## 4. 核心算法设计

### 4.1 xLSTM模块设计

#### 4.1.1 mLSTM（矩阵记忆LSTM）

**数学定义**：

```
输入: x_t ∈ ℝ^D, h_(t-1) ∈ ℝ^D, c_(t-1) ∈ ℝ^D, M_(t-1) ∈ ℝ^(D×D)

门控计算:
  [i_t, f_t, z_t, o_t] = [W_i, W_f, W_z, W_o] · [x_t; h_(t-1)] + [b_i, b_f, b_z, b_o]
  i_t = σ(i_t)    # 输入门
  f_t = σ(f_t)    # 遗忘门
  z_t = tanh(z_t) # 输入内容
  o_t = σ(o_t)    # 输出门

矩阵记忆更新:
  M_t = f_t ⊙ M_(t-1) + i_t ⊙ (z_t ⊗ z_t)  # 外积更新矩阵记忆
  
状态归一化:
  c_t = f_t ⊙ c_(t-1) + i_t ⊙ z_t
  h_t = o_t ⊙ tanh(Normalize(M_t · c_t))

输出: h_t, c_t, M_t
```

**关键特性**：
- 矩阵记忆M_t存储长期依赖关系
- 外积更新机制增强记忆容量
- 状态归一化稳定训练

#### 4.1.2 sLSTM（标量LSTM）

**数学定义**：

```
输入: x_t ∈ ℝ^D, h_(t-1) ∈ ℝ^D, c_(t-1) ∈ ℝ^D, n_(t-1) ∈ ℝ^D

门控计算:
  [i_t, f_t, z_t, o_t] = [W_i, W_f, W_z, W_o] · [x_t; h_(t-1)] + [b_i, b_f, b_z, b_o]
  
指数门控:
  i_t = exp(i_t)    # 指数输入门
  f_t = exp(f_t)    # 指数遗忘门
  z_t = tanh(z_t)   # 输入内容
  o_t = σ(o_t)      # 输出门

状态更新:
  c_t = f_t ⊙ c_(t-1) + i_t ⊙ z_t
  n_t = f_t ⊙ n_(t-1) + i_t           # 归一化状态
  h_t = o_t ⊙ tanh(c_t / n_t)         # 归一化输出

输出: h_t, c_t, n_t
```

**关键特性**：
- 指数门控机制增强非线性表达能力
- 归一化状态解决梯度消失问题
- 标量运算效率高

#### 4.1.3 xLSTM Block结构

```
输入: X ∈ ℝ^(B×L×D)

并行计算:
  H_m = mLSTM(X)    # 矩阵记忆分支
  H_s = sLSTM(X)    # 标量门控分支

门控融合:
  g_m = σ(Linear([H_m; H_s]))  # 矩阵分支门控
  g_s = 1 - g_m                 # 标量分支门控
  H_out = g_m ⊙ H_m + g_s ⊙ H_s

输出: H_out ∈ ℝ^(B×L×D)
```

### 4.2 模型训练策略

#### 4.2.1 损失函数

**主要损失**：
```python
# Huber Loss - 对异常值更鲁棒
def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    is_small_error = tf.abs(error) <= delta
    small_error_loss = 0.5 * tf.square(error)
    big_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, small_error_loss, big_error_loss)

# 总损失
loss = huber_loss(y_pred, y_true) + λ * regularization_loss
```

**辅助损失（可选）**：
- 特征一致性损失：保证xLSTM和Transformer特征分布一致
- 时序平滑损失：保证相邻时间步预测平滑

#### 4.2.2 优化器配置

| 参数 | 值 | 说明 |
|-----|---|------|
| 优化器 | AdamW | 权重衰减解耦 |
| 学习率 | 1e-4 | 初始学习率 |
| 学习率调度 | CosineAnnealingWarmRestarts | 余弦退火重启 |
| 权重衰减 | 0.01 | L2正则化系数 |
| 梯度裁剪 | 1.0 | 防止梯度爆炸 |
| Batch Size | 32 | 根据显存调整 |
| Epochs | 200 | 早停机制 |

#### 4.2.3 早停与检查点

```python
# 早停配置
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,          # 20轮无改善则停止
    restore_best_weights=True
)

# 检查点配置
checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_{epoch:03d}_{val_loss:.4f}.pt',
    monitor='val_loss',
    save_best_only=True,
    save_freq='epoch'
)
```

---

## 5. 数据库设计

### 5.1 关系型数据库设计（SQLite）

#### 5.1.1 数据库ER图

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   users     │       │  batteries  │       │ cycle_data  │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ PK id       │       │ PK id       │◀──────┤ FK battery_id│
│ username    │       │ name        │       │ PK id       │
│ email       │       │ type        │       │ cycle_number│
│ password_hash│      │ rated_cap   │       │ capacity    │
│ role        │       │ nominal_volt│       │ voltage_max │
│ created_at  │       │ created_at  │       │ ...         │
└─────────────┘       └─────────────┘       └─────────────┘
         │
         │            ┌─────────────┐       ┌─────────────┐
         │            │ predictions │       │  diagnoses  │
         └───────────▶├─────────────┤       ├─────────────┤
              FK user_id            │◀──────┤ FK battery_id│
                       PK id        │       │ PK id       │
                       FK battery_id│       │ fault_type  │
                       cycle_number │       │ severity    │
                       predicted_rul│       │ report      │
                       confidence   │       │ created_at  │
                       created_at   │       └─────────────┘
                       └─────────────┘
```

#### 5.1.2 表结构设计

**表1: users（用户表）**

| 字段名 | 类型 | 约束 | 默认值 | 说明 |
|-------|------|------|--------|------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | - | 用户ID |
| username | VARCHAR(50) | UNIQUE NOT NULL | - | 用户名 |
| email | VARCHAR(100) | UNIQUE NOT NULL | - | 邮箱 |
| password_hash | VARCHAR(255) | NOT NULL | - | 密码哈希(bcrypt) |
| role | VARCHAR(20) | DEFAULT 'user' | 'user' | 角色(user/admin) |
| is_active | BOOLEAN | DEFAULT 1 | 1 | 是否激活 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | - | 创建时间 |
| updated_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | - | 更新时间 |

**表2: batteries（电池信息表）**

| 字段名 | 类型 | 约束 | 默认值 | 说明 |
|-------|------|------|--------|------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | - | 电池ID |
| name | VARCHAR(100) | NOT NULL | - | 电池名称 |
| type | VARCHAR(50) | | - | 电池型号 |
| rated_capacity | FLOAT | | - | 额定容量(Ah) |
| nominal_voltage | FLOAT | | - | 标称电压(V) |
| chemistry | VARCHAR(50) | | 'LFP' | 电池化学类型 |
| manufacturer | VARCHAR(100) | | - | 制造商 |
| current_cycle | INTEGER | DEFAULT 0 | 0 | 当前循环次数 |
| current_soh | FLOAT | | - | 当前健康状态(%) |
| status | VARCHAR(20) | DEFAULT 'active' | 'active' | 状态 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | - | 创建时间 |
| updated_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | - | 更新时间 |

**表3: cycle_data（循环数据表）**

| 字段名 | 类型 | 约束 | 默认值 | 说明 |
|-------|------|------|--------|------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | - | 数据ID |
| battery_id | INTEGER | FOREIGN KEY | - | 关联电池ID |
| cycle_number | INTEGER | NOT NULL | - | 循环次数 |
| capacity | FLOAT | | - | 放电容量(Ah) |
| voltage_max | FLOAT | | - | 最大电压(V) |
| voltage_min | FLOAT | | - | 最小电压(V) |
| temperature_max | FLOAT | | - | 最高温度(℃) |
| temperature_min | FLOAT | | - | 最低温度(℃) |
| charge_time | FLOAT | | - | 充电时间(h) |
| discharge_time | FLOAT | | - | 放电时间(h) |
| energy | FLOAT | | - | 放电能量(Wh) |
| coulombic_efficiency | FLOAT | | - | 库伦效率(%) |
| internal_resistance | FLOAT | | - | 内阻(mΩ) |
| record_time | DATETIME | | - | 记录时间 |

**表4: predictions（预测结果表）**

| 字段名 | 类型 | 约束 | 默认值 | 说明 |
|-------|------|------|--------|------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | - | 预测ID |
| battery_id | INTEGER | FOREIGN KEY | - | 关联电池ID |
| user_id | INTEGER | FOREIGN KEY | - | 关联用户ID |
| cycle_number | INTEGER | | - | 当前循环次数 |
| predicted_rul | FLOAT | | - | 预测剩余寿命(循环数) |
| actual_rul | FLOAT | | - | 实际剩余寿命 |
| confidence | FLOAT | | - | 预测置信度 |
| model_version | VARCHAR(20) | | - | 模型版本 |
| prediction_time | DATETIME | DEFAULT CURRENT_TIMESTAMP | - | 预测时间 |

**表5: diagnoses（诊断记录表）**

| 字段名 | 类型 | 约束 | 默认值 | 说明 |
|-------|------|------|--------|------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | - | 诊断ID |
| battery_id | INTEGER | FOREIGN KEY | - | 关联电池ID |
| fault_type | VARCHAR(100) | | - | 故障类型 |
| severity | VARCHAR(20) | | - | 严重程度 |
| symptoms | TEXT | | - | 症状列表(JSON) |
| causes | TEXT | | - | 原因列表(JSON) |
| solutions | TEXT | | - | 解决方案(JSON) |
| report | TEXT | | - | 诊断报告 |
| confidence | FLOAT | | - | 置信度 |
| diagnosis_time | DATETIME | DEFAULT CURRENT_TIMESTAMP | - | 诊断时间 |

### 5.2 知识图谱设计（Neo4j）

#### 5.2.1 节点类型

```cypher
// 电池节点
CREATE (:Battery {
  id: 'B001',
  type: '18650',
  chemistry: 'NMC',
  rated_capacity: 3.0
})

// 症状节点
CREATE (:Symptom {
  name: '容量骤降',
  description: '容量在短时间内快速下降',
  threshold: 0.2
})

// 故障节点
CREATE (:Fault {
  name: '析锂',
  mechanism: '锂离子在负极表面还原为金属锂',
  severity: 'high'
})

// 原因节点
CREATE (:Cause {
  name: '过充',
  description: '充电电压超过上限',
  probability: 0.8
})

// 解决方案节点
CREATE (:Solution {
  action: '降低充电截止电压',
  priority: 1,
  reference: '操作手册3.2节'
})
```

#### 5.2.2 关系类型

```cypher
// 电池-症状关系
CREATE (b:Battery)-[:HAS_SYMPTOM {
  detected_at: '2026-03-10',
  value: 0.25
}]->(s:Symptom)

// 症状-故障关系
CREATE (s:Symptom)-[:INDICATES {
  confidence: 0.85
}]->(f:Fault)

// 故障-原因关系
CREATE (f:Fault)-[:CAUSED_BY {
  probability: 0.7
}]->(c:Cause)

// 故障-解决方案关系
CREATE (f:Fault)-[:HAS_SOLUTION {
  effectiveness: 0.9
}]->(sol:Solution)
```

---

## 6. 接口设计

### 6.1 RESTful API设计规范

#### 6.1.1 API版本管理

```
/api/v1/              # 版本1 API
/api/v2/              # 版本2 API（未来）
```

#### 6.1.2 统一响应格式

```json
{
  "code": 200,
  "message": "success",
  "data": {
    // 业务数据
  },
  "timestamp": "2026-03-10T10:00:00Z"
}
```

#### 6.1.3 分页响应格式

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [],
    "total": 100,
    "page": 1,
    "page_size": 20,
    "total_pages": 5
  },
  "timestamp": "2026-03-10T10:00:00Z"
}
```

### 6.2 核心API接口

#### 6.2.1 数据管理API

| 接口 | 方法 | 路径 | 说明 |
|-----|------|------|------|
| 获取电池列表 | GET | /api/v1/data/batteries | 分页查询电池列表 |
| 获取电池详情 | GET | /api/v1/data/batteries/{id} | 查询单个电池信息 |
| 上传电池数据 | POST | /api/v1/data/upload | 上传电池数据文件 |
| 获取循环数据 | GET | /api/v1/data/batteries/{id}/cycles | 查询循环数据 |

#### 6.2.2 预测服务API

| 接口 | 方法 | 路径 | 说明 |
|-----|------|------|------|
| RUL预测 | POST | /api/v1/prediction/rul | 执行RUL预测 |
| 批量预测 | POST | /api/v1/prediction/batch | 批量RUL预测 |
| 预测历史 | GET | /api/v1/prediction/history/{battery_id} | 查询预测历史 |

#### 6.2.3 诊断服务API

| 接口 | 方法 | 路径 | 说明 |
|-----|------|------|------|
| 故障诊断 | POST | /api/v1/diagnosis/analyze | 执行故障诊断 |
| 诊断历史 | GET | /api/v1/diagnosis/history/{battery_id} | 查询诊断历史 |
| 诊断报告 | GET | /api/v1/diagnosis/report/{id} | 获取诊断报告 |

---

## 7. 安全设计

### 7.1 认证与授权

#### 7.1.1 JWT认证

```python
# JWT配置
JWT_SECRET_KEY = "your-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 1440  # 24小时

# Token生成
def create_access_token(user_id: int):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
```

#### 7.1.2 权限控制

| 角色 | 权限 |
|-----|------|
| admin | 所有权限 |
| user | 数据查看、预测、诊断 |
| viewer | 仅查看 |

### 7.2 数据安全

#### 7.2.1 密码加密

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 密码哈希
hashed_password = pwd_context.hash(plain_password)

# 密码验证
is_valid = pwd_context.verify(plain_password, hashed_password)
```

#### 7.2.2 敏感数据保护

- 数据库连接字符串加密存储
- API密钥使用环境变量
- 日志脱敏处理

### 7.3 接口安全

#### 7.3.1 请求限流

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/prediction/rul")
@limiter.limit("10/minute")  # 每分钟最多10次
async def predict_rul(request: Request):
    pass
```

#### 7.3.2 输入验证

```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    battery_id: int
    
    @validator('battery_id')
    def validate_battery_id(cls, v):
        if v <= 0:
            raise ValueError('battery_id must be positive')
        return v
```

---

## 8. 性能设计

### 8.1 性能优化策略

#### 8.1.1 模型推理优化

| 优化项 | 方法 | 预期效果 |
|-------|------|---------|
| 模型量化 | INT8量化 | 推理速度提升2-3倍 |
| 批处理 | 动态批处理 | 吞吐量提升50% |
| 模型缓存 | Redis缓存 | 重复请求响应<10ms |
| 异步处理 | Celery任务队列 | 并发能力提升 |

#### 8.1.2 数据库优化

| 优化项 | 方法 | 说明 |
|-------|------|------|
| 索引优化 | 关键字段建索引 | 加速查询 |
| 连接池 | SQLAlchemy连接池 | 复用连接 |
| 查询优化 | 避免N+1查询 | 减少数据库访问 |
| 分页查询 | LIMIT + OFFSET | 大数据集分页 |

#### 8.1.3 前端优化

| 优化项 | 方法 | 说明 |
|-------|------|------|
| 代码分割 | React.lazy | 按需加载 |
| 图表优化 | ECharts数据采样 | 大数据量渲染 |
| 缓存策略 | React Query缓存 | 减少API请求 |
| 虚拟滚动 | react-window | 长列表优化 |

### 8.2 性能监控

#### 8.2.1 监控指标

| 指标类型 | 具体指标 | 阈值 |
|---------|---------|------|
| API性能 | 响应时间 | P99 < 500ms |
| 模型性能 | 推理延迟 | < 200ms |
| 系统资源 | CPU使用率 | < 80% |
| 系统资源 | 内存使用率 | < 85% |
| 错误率 | 5xx错误率 | < 0.1% |

#### 8.2.2 监控工具

- Prometheus：指标采集
- Grafana：可视化监控
- ELK Stack：日志分析

---

**文档结束**
