# counterfactual_paired 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：750
- 总耗时：1538.552 秒
- 峰值显存：3.123 GiB
- 预测 SHA-256：`7df9f7a763b0216729165a1784bc4877312745bb20984576797a454a20679287`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 150 | 76.67% | 76.48% | 75.33% | 75.33% | 75.33% |
| capability_substitution | 150 | 0.67% | 0.67% | 0.67% | 0.67% | 0.67% |
| distractor | 150 | 92.67% | 86.66% | 74.00% | 74.00% | 74.00% |
| ephemeral_interface | 150 | 84.67% | 78.83% | 67.33% | 67.33% | 67.33% |
| original | 150 | 94.00% | 88.46% | 76.00% | 76.00% | 76.00% |

## 条件化指标

- Interface Equivariance Accuracy：67.33%
- All-worlds Group Accuracy：0.67%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 59.33% | 89 / 150 |
| original + capability_substitution | 0.67% | 1 / 150 |
| original + distractor | 70.67% | 106 / 150 |
| original + ephemeral_interface | 62.67% | 94 / 150 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
