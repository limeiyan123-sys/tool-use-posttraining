# Stage 1 Zero-shot 与 LoRA SFT 配对比较

> 对同一冻结测试集逐样本比较；95% CI 由 10000 次配对 bootstrap 得到。

## 总体指标

| 指标 | Zero-shot | LoRA SFT | 差值 | 95% CI |
|---|---:|---:|---:|---:|
| tool_accuracy | 95.71% | 100.00% | +4.29% | [+0.00%, +10.00%] |
| argument_accuracy | 90.25% | 93.94% | +3.69% | [-1.19%, +9.52%] |
| argument_exact_match | 80.00% | 81.43% | +1.43% | [-7.14%, +10.00%] |

## 分轨比较

| 轨道 | 样本数 | Tool 差值 | Arg 差值 | Arg Exact 差值 |
|---|---:|---:|---:|---:|
| bfcl_holdout_v1 | 70 | +4.29% | +3.69% | +1.43% |

## 错误迁移

- `argument_error -> argument_error`：9
- `argument_error -> none`：2
- `none -> argument_error`：4
- `none -> none`：52
- `tool_selection_error -> none`：3

## 解释边界

- 置信区间反映当前 98 条冻结样本上的抽样不确定性，不代表 BFCL 官方总体。
- 若区间跨过 0，只能报告观察到的趋势，不能宣称统计显著改进。
