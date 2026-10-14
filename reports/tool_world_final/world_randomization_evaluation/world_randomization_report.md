# world_randomization 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：300
- 总耗时：209.791 秒
- 峰值显存：3.04 GiB
- 预测 SHA-256：`e89b50f9bb0f0729bbb706a7c9294ed3b7b04c07a001b4a76d09770d60e8b70b`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 60 | 83.33% | 82.64% | 80.00% | 80.00% | 80.00% |
| capability_substitution | 60 | 3.33% | 3.33% | 3.33% | 3.33% | 3.33% |
| distractor | 60 | 95.00% | 85.25% | 65.00% | 65.00% | 65.00% |
| ephemeral_interface | 60 | 86.67% | 74.64% | 56.67% | 56.67% | 56.67% |
| original | 60 | 96.67% | 88.63% | 71.67% | 71.67% | 71.67% |

## 条件化指标

- Interface Equivariance Accuracy：56.67%
- All-worlds Group Accuracy：1.67%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 63.33% | 38 / 60 |
| original + capability_substitution | 1.67% | 1 / 60 |
| original + distractor | 61.67% | 37 / 60 |
| original + ephemeral_interface | 56.67% | 34 / 60 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据为方法冻结后一次性使用的独立最终测试集，不再据此修改方法。
