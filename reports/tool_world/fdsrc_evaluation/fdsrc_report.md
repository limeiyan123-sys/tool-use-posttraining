# fdsrc 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：750
- 总耗时：2045.569 秒
- 峰值显存：3.123 GiB
- 预测 SHA-256：`fd32d128bf58c4aacd472774b87883d3f2f7f04a22f876253306b09cd091624e`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 150 | 75.33% | 75.22% | 74.67% | 74.67% | 74.67% |
| capability_substitution | 150 | 0.67% | 0.67% | 0.67% | 0.67% | 0.67% |
| distractor | 150 | 78.67% | 72.20% | 62.00% | 62.00% | 62.00% |
| ephemeral_interface | 150 | 84.00% | 75.31% | 61.33% | 61.33% | 61.33% |
| original | 150 | 93.33% | 87.06% | 74.67% | 74.67% | 74.67% |

## 条件化指标

- Interface Equivariance Accuracy：61.33%
- All-worlds Group Accuracy：0.67%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 58.00% | 87 / 150 |
| original + capability_substitution | 0.67% | 1 / 150 |
| original + distractor | 58.67% | 88 / 150 |
| original + ephemeral_interface | 57.33% | 86 / 150 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
