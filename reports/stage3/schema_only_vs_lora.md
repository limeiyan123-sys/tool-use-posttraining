# Stage 1 Zero-shot 与 LoRA SFT 配对比较

> 对同一冻结测试集逐样本比较；95% CI 由 10000 次配对 bootstrap 得到。

## 总体指标

| 指标 | Zero-shot | LoRA SFT | 差值 | 95% CI |
|---|---:|---:|---:|---:|
| tool_accuracy | 93.88% | 93.88% | +0.00% | [-3.06%, +3.06%] |
| argument_accuracy | 88.75% | 88.83% | +0.09% | [-2.72%, +3.06%] |
| argument_exact_match | 80.61% | 79.59% | -1.02% | [-5.10%, +2.04%] |

## 分轨比较

| 轨道 | 样本数 | Tool 差值 | Arg 差值 | Arg Exact 差值 |
|---|---:|---:|---:|---:|
| bfcl_lite_v1 | 70 | -1.43% | -1.31% | -2.86% |
| controlled_v1 | 28 | +3.57% | +3.57% | +3.57% |

## 错误迁移

- `argument_error -> argument_error`：12
- `argument_error -> parse_error`：1
- `none -> argument_error`：2
- `none -> none`：76
- `schema_validation_error -> schema_validation_error`：1
- `tool_selection_error -> none`：1
- `tool_selection_error -> tool_selection_error`：5

## 解释边界

- 置信区间反映当前 98 条冻结样本上的抽样不确定性，不代表 BFCL 官方总体。
- 若区间跨过 0，只能报告观察到的趋势，不能宣称统计显著改进。
