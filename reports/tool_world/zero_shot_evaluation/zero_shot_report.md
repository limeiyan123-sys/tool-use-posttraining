# zero_shot 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：750
- 总耗时：757.99 秒
- 峰值显存：2.988 GiB
- 预测 SHA-256：`2a120cd0eda3e4fa4228ac72888dfedc815f2b1f9b38446dcdbd8a5c563f7128`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 150 | 67.33% | 67.26% | 66.67% | 66.67% | 66.67% |
| capability_substitution | 150 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| distractor | 150 | 88.00% | 80.16% | 66.67% | 66.67% | 66.67% |
| ephemeral_interface | 150 | 86.67% | 65.29% | 48.67% | 48.67% | 48.67% |
| original | 150 | 94.67% | 86.23% | 71.33% | 71.33% | 71.33% |

## 条件化指标

- Interface Equivariance Accuracy：48.67%
- All-worlds Group Accuracy：0.00%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 48.00% | 72 / 150 |
| original + capability_substitution | 0.00% | 0 / 150 |
| original + distractor | 61.33% | 92 / 150 |
| original + ephemeral_interface | 43.33% | 65 / 150 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
