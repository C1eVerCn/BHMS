# BHMS 案例包

## 一、样本概况
- 电池 ID：kaggle::kaggle_demo::KAGGLE_UNSEEN_201
- 数据来源：KAGGLE
- 数据集：kaggle_demo
- 循环次数：30
- 当前健康分：80.99
- 当前容量：1.7300Ah

## 二、数据画像
- 当前来源样本数：7
- 当前来源周期点：510
- 当前来源训练候选：0
- 当前来源数据集：kaggle_demo
- 当前样本划分：unassigned
- 当前样本训练池状态：未加入

## 三、历史轨迹摘要
- 已读取 cycle 点：30
- 首个 cycle：1，容量 2.1360Ah
- 最新 cycle：30，容量 1.7300Ah

## 四、RUL 预测摘要
- 模型：hybrid
- 预测 RUL：2.57 cycles
- 置信度：97.0%
- Checkpoint：hybrid_best.pt
- 关键特征 temperature_rise_rate: 遮挡 temperature_rise_rate 后，模型输出相对基线提升了 RUL 估计。
- 关键特征 cycle_number: 遮挡 cycle_number 后，模型输出相对基线压低了 RUL 估计。
- 关键特征 voltage_mean: 遮挡 voltage_mean 后，模型输出相对基线压低了 RUL 估计。
- 关键特征 capacity: 遮挡 capacity 后，模型输出相对基线提升了 RUL 估计。

## 五、异常与诊断
- 异常症状：内阻异常 / 内阻增加至基线的 147.5% 。
- 诊断结论：内阻增大
- 诊断置信度：59.7%
- 根因：SEI膜增厚
- 根因：长期存储
- 根因：低温循环
- 建议：进行容量校准
- 建议：优化工作温度
- 建议：降低峰值电流

## 六、实验背景
- 当前来源对比结果中表现较优的模型：bilstm
- 实验风险：bilstm 的 R2 仍为负值，说明实验结论更偏演示级。
- 实验风险：bilstm 的 MAPE 偏高，建议补充多随机种子与消融实验。
- 实验风险：hybrid 的 R2 仍为负值，说明实验结论更偏演示级。

## 七、答辩讲解建议
- 先用样本概况回答“数据来自哪里”。
- 再用 RUL 预测回答“电池还能用多久”。
- 用异常与诊断链回答“为什么会变差、应该怎么处理”。
- 最后用实验背景说明“为什么当前模型设计值得做，但还有哪些局限”。