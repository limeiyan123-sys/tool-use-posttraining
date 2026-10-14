# Stage 1 Zero-shot 与 LoRA SFT 配对比较

> 对同一冻结测试集逐样本比较；95% CI 由 10000 次配对 bootstrap 得到。

## 总体指标

| 指标 | Zero-shot | LoRA SFT | 差值 | 95% CI |
|---|---:|---:|---:|---:|
| tool_accuracy | 91.84% | 93.88% | +2.04% | [-2.04%, +7.14%] |
| argument_accuracy | 86.90% | 88.75% | +1.84% | [-3.06%, +6.92%] |
| argument_exact_match | 76.53% | 80.61% | +4.08% | [-1.02%, +10.20%] |

## 分轨比较

| 轨道 | 样本数 | Tool 差值 | Arg 差值 | Arg Exact 差值 |
|---|---:|---:|---:|---:|
| bfcl_lite_v1 | 70 | +2.86% | +2.58% | +4.29% |
| controlled_v1 | 28 | +0.00% | +0.00% | +3.57% |

## 错误迁移

- `argument_error -> argument_error`：12
- `argument_error -> none`：3
- `none -> none`：72
- `none -> tool_selection_error`：2
- `parse_error -> argument_error`：1
- `schema_validation_error -> schema_validation_error`：1
- `tool_selection_error -> none`：3
- `tool_selection_error -> tool_selection_error`：4

## 解释边界

- 置信区间反映当前 98 条冻结样本上的抽样不确定性，不代表 BFCL 官方总体。
- 若区间跨过 0，只能报告观察到的趋势，不能宣称统计显著改进。
