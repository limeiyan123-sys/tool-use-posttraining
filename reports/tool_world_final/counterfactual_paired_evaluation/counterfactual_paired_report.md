# counterfactual_paired 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：300
- 总耗时：186.987 秒
- 峰值显存：3.04 GiB
- 预测 SHA-256：`6c768bb60c06d7a2034aaf561b6c3b37ab4d430266b36a83cfdf363e44cb1cd7`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 60 | 83.33% | 82.64% | 80.00% | 80.00% | 80.00% |
| capability_substitution | 60 | 1.67% | 1.67% | 1.67% | 1.67% | 1.67% |
| distractor | 60 | 91.67% | 82.75% | 66.67% | 66.67% | 66.67% |
| ephemeral_interface | 60 | 90.00% | 76.81% | 53.33% | 53.33% | 53.33% |
| original | 60 | 95.00% | 85.71% | 68.33% | 68.33% | 68.33% |

## 条件化指标

- Interface Equivariance Accuracy：53.33%
- All-worlds Group Accuracy：0.00%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 60.00% | 36 / 60 |
| original + capability_substitution | 1.67% | 1 / 60 |
| original + distractor | 58.33% | 35 / 60 |
| original + ephemeral_interface | 51.67% | 31 / 60 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据为方法冻结后一次性使用的独立最终测试集，不再据此修改方法。
