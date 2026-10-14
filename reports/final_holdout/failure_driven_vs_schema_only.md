# Stage 1 Zero-shot 与 LoRA SFT 配对比较

> 对同一冻结测试集逐样本比较；95% CI 由 10000 次配对 bootstrap 得到。

## 总体指标

| 指标 | Zero-shot | LoRA SFT | 差值 | 95% CI |
|---|---:|---:|---:|---:|
| tool_accuracy | 98.57% | 100.00% | +1.43% | [+0.00%, +4.29%] |
| argument_accuracy | 91.56% | 93.94% | +2.38% | [-0.95%, +6.43%] |
| argument_exact_match | 77.14% | 81.43% | +4.29% | [-1.43%, +11.43%] |

## 分轨比较

| 轨道 | 样本数 | Tool 差值 | Arg 差值 | Arg Exact 差值 |
|---|---:|---:|---:|---:|
| bfcl_holdout_v1 | 70 | +1.43% | +2.38% | +4.29% |

## 错误迁移

- `argument_error -> argument_error`：12
- `argument_error -> none`：3
- `none -> argument_error`：1
- `none -> none`：53
- `tool_selection_error -> none`：1

## 解释边界

- 置信区间反映当前 98 条冻结样本上的抽样不确定性，不代表 BFCL 官方总体。
- 若区间跨过 0，只能报告观察到的趋势，不能宣称统计显著改进。
