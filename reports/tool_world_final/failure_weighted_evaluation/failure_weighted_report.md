# failure_weighted 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：300
- 总耗时：54.723 秒
- 峰值显存：3.039 GiB
- 预测 SHA-256：`5755b0aaf553eda6871519913b5008204d65c6164335673ea0096520e4dd3b8e`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 60 | 83.33% | 82.64% | 80.00% | 80.00% | 80.00% |
| capability_substitution | 60 | 56.67% | 56.67% | 56.67% | 56.67% | 56.67% |
| distractor | 60 | 71.67% | 66.41% | 53.33% | 53.33% | 53.33% |
| ephemeral_interface | 60 | 86.67% | 73.00% | 53.33% | 51.67% | 53.33% |
| original | 60 | 95.00% | 86.69% | 70.00% | 70.00% | 70.00% |

## 条件化指标

- Interface Equivariance Accuracy：53.33%
- All-worlds Group Accuracy：18.33%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 61.67% | 37 / 60 |
| original + capability_substitution | 41.67% | 25 / 60 |
| original + distractor | 50.00% | 30 / 60 |
| original + ephemeral_interface | 50.00% | 30 / 60 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据为方法冻结后一次性使用的独立最终测试集，不再据此修改方法。
