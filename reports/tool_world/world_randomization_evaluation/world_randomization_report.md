# world_randomization 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：750
- 总耗时：1583.247 秒
- 峰值显存：3.123 GiB
- 预测 SHA-256：`602f2144d2a46bd88b49c3239b85fa9bc8467c65cc2e5d2399554725a1aa39d5`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 150 | 77.33% | 77.22% | 76.67% | 76.67% | 76.67% |
| capability_substitution | 150 | 3.33% | 3.33% | 3.33% | 3.33% | 3.33% |
| distractor | 150 | 90.67% | 84.09% | 73.33% | 73.33% | 73.33% |
| ephemeral_interface | 150 | 84.67% | 80.11% | 70.67% | 70.67% | 70.67% |
| original | 150 | 94.00% | 88.59% | 76.00% | 76.00% | 76.00% |

## 条件化指标

- Interface Equivariance Accuracy：70.67%
- All-worlds Group Accuracy：2.67%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 60.67% | 91 / 150 |
| original + capability_substitution | 2.67% | 4 / 150 |
| original + distractor | 70.00% | 105 / 150 |
| original + ephemeral_interface | 66.67% | 100 / 150 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
