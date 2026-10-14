# Stage 1 Zero-shot 与 LoRA SFT 配对比较

> 对同一冻结测试集逐样本比较；95% CI 由 10000 次配对 bootstrap 得到。

## 总体指标

| 指标 | Zero-shot | LoRA SFT | 差值 | 95% CI |
|---|---:|---:|---:|---:|
| tool_accuracy | 97.14% | 100.00% | +2.86% | [+0.00%, +7.14%] |
| argument_accuracy | 91.14% | 93.94% | +2.80% | [-0.71%, +7.14%] |
| argument_exact_match | 78.57% | 81.43% | +2.86% | [-2.86%, +8.57%] |

## 分轨比较

| 轨道 | 样本数 | Tool 差值 | Arg 差值 | Arg Exact 差值 |
|---|---:|---:|---:|---:|
| bfcl_holdout_v1 | 70 | +2.86% | +2.80% | +2.86% |

## 错误迁移

- `argument_error -> argument_error`：11
- `argument_error -> none`：2
- `none -> argument_error`：1
- `none -> none`：54
- `tool_selection_error -> argument_error`：1
- `tool_selection_error -> none`：1

## 解释边界

- 置信区间反映当前 98 条冻结样本上的抽样不确定性，不代表 BFCL 官方总体。
- 若区间跨过 0，只能报告观察到的趋势，不能宣称统计显著改进。
