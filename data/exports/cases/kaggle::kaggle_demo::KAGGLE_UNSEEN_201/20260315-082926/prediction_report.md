# 电池寿命预测报告

## 一、预测概览
- 电池 ID：kaggle::kaggle_demo::KAGGLE_UNSEEN_201
- 数据来源：kaggle
- 模型：hybrid
- 预测 RUL：2.57 cycles
- 置信度：97.0%
- 预测时间：2026-03-15T02:50:34.999807

## 二、寿命轨迹解释
- 当前历史轨迹长度：30 个 cycle
- 预测 EOL 周期：32.57
- EOL 容量阈值：1.7088
- 投影方法：linear

## 三、模型证据链
- 特征 temperature_rise_rate：影响值 0.004，遮挡 temperature_rise_rate 后，模型输出相对基线提升了 RUL 估计。
- 特征 cycle_number：影响值 0.0013，遮挡 cycle_number 后，模型输出相对基线压低了 RUL 估计。
- 特征 voltage_mean：影响值 0.0012，遮挡 voltage_mean 后，模型输出相对基线压低了 RUL 估计。
- 特征 capacity：影响值 0.0006，遮挡 capacity 后，模型输出相对基线提升了 RUL 估计。
- 特征 current_mean：影响值 0.0004，遮挡 current_mean 后，模型输出相对基线压低了 RUL 估计。

## 四、关键时间窗口
- cycles 1-7：影响值 0.0056，遮挡该时间窗口后，预测 RUL 变化 0.006。
- cycles 23-30：影响值 0.0043，遮挡该时间窗口后，预测 RUL 变化 0.004。
- cycles 8-15：影响值 0.0017，遮挡该时间窗口后，预测 RUL 变化 0.002。
- cycles 16-22：影响值 0.0009，遮挡该时间窗口后，预测 RUL 变化 0.001。

## 五、置信度说明
- 输入窗口长度为 30 个 cycle
- 当前模型为 hybrid，来源 kaggle
- 本次预测直接使用训练权重推理
- 轨迹稳定度评分为 1.000

## 六、限制说明
- 未来容量曲线属于受 RUL 约束的可视化投影，不等同于序列生成模型直接输出。
- 当前系统展示的是可审计预测证据链，而不是模型内部隐式思维过程。