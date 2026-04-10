# BHMS 论文证据包

- 生成时间：2026-04-09T05:54:41.781620
- 总体论文门槛：未通过
- 真值来源：仅使用 `*_multi_seed_summary.json` 与 `*_transfer_summary.json` 聚合结果。

## 核心 benchmark 矩阵

| Source | Unit | Hybrid RMSE | Hybrid R2 | BiLSTM RMSE | BiLSTM R2 | Winner | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nasa | within_source | 0.30343 | -1.123352 | 0.304044 | -1.124386 | hybrid | pass |
| nasa | transfer | 0.303572 | -1.11717 | 0.304276 | -1.134913 | hybrid | pass |
| calce | within_source | 0.021945 | 0.513922 | 0.017631 | 0.695668 | bilstm | fail |
| calce | transfer | 0.018598 | 0.657456 | 0.024387 | 0.412059 | hybrid | pass |
| kaggle | within_source | 0.027337 | 0.322002 | 0.02402 | 0.512857 | bilstm | fail |
| hust | within_source | 0.010682 | 0.97475 | 0.014497 | 0.953587 | hybrid | pass |
| matr | within_source | 0.088858 | 0.437311 | 0.09231 | 0.390116 | hybrid | pass |

## 结论

- 当前核心矩阵尚未满足 Hybrid 在全部 benchmark 单元上全面优于 BiLSTM，论文主结论仍需保持“继续优化中”。
- 未通过单元：nasa / ablation，存在 blocking 变体：no_xlstm, no_transformer。
- 未通过单元：calce / within_source，Hybrid RMSE=0.021945，BiLSTM RMSE=0.017631，Hybrid R2=0.513922，BiLSTM R2=0.695668。
- 未通过单元：calce / ablation，存在 blocking 变体：no_xlstm, no_transformer。
- 未通过单元：kaggle / within_source，Hybrid RMSE=0.027337，BiLSTM RMSE=0.02402，Hybrid R2=0.322002，BiLSTM R2=0.512857。
- 未通过单元：matr / ablation，存在 blocking 变体：no_transformer。

## 消融门槛

- nasa: fail
  - blocking: no_xlstm rmse=0.303318 r2=-1.119997
  - blocking: no_transformer rmse=0.30312 r2=-1.11695
- calce: fail
  - blocking: no_xlstm rmse=0.014203 r2=0.800341
  - blocking: no_transformer rmse=0.014212 r2=0.802236
- kaggle: pass
- hust: pass
- matr: fail
  - blocking: no_transformer rmse=0.084792 r2=0.491305