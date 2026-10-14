# schema_only 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：750
- 总耗时：2006.542 秒
- 峰值显存：3.123 GiB
- 预测 SHA-256：`19661e42271ac752f1f809e9d6c49c8778f64c7fc3d10bcec68148f0da36f050`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 150 | 76.67% | 76.30% | 75.33% | 75.33% | 75.33% |
| capability_substitution | 150 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| distractor | 150 | 89.33% | 81.76% | 68.00% | 68.00% | 68.00% |
| ephemeral_interface | 150 | 86.00% | 77.96% | 63.33% | 63.33% | 63.33% |
| original | 150 | 94.67% | 86.70% | 72.67% | 72.67% | 72.67% |

## 条件化指标

- Interface Equivariance Accuracy：63.33%
- All-worlds Group Accuracy：0.00%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 58.00% | 87 / 150 |
| original + capability_substitution | 0.00% | 0 / 150 |
| original + distractor | 65.33% | 98 / 150 |
| original + ephemeral_interface | 60.00% | 90 / 150 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
