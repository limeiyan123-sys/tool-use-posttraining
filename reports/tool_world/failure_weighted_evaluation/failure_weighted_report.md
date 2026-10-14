# failure_weighted 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：750
- 总耗时：1311.282 秒
- 峰值显存：3.123 GiB
- 预测 SHA-256：`101b86abe434965150b801d1b632f3feda40a26271a07a79f6df80b8291993d6`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 150 | 78.67% | 78.33% | 77.33% | 77.33% | 77.33% |
| capability_substitution | 150 | 47.33% | 47.33% | 47.33% | 47.33% | 47.33% |
| distractor | 150 | 74.67% | 68.44% | 58.67% | 58.67% | 58.67% |
| ephemeral_interface | 150 | 81.33% | 75.22% | 64.67% | 64.67% | 64.67% |
| original | 150 | 90.00% | 84.50% | 72.67% | 72.67% | 72.67% |

## 条件化指标

- Interface Equivariance Accuracy：64.67%
- All-worlds Group Accuracy：14.00%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 58.67% | 88 / 150 |
| original + capability_substitution | 34.67% | 52 / 150 |
| original + distractor | 56.00% | 84 / 150 |
| original + ephemeral_interface | 60.67% | 91 / 150 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
