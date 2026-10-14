# lora 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：750
- 总耗时：1966.191 秒
- 峰值显存：3.123 GiB
- 预测 SHA-256：`a0115649ff422943b96091d3257c222bad5972c477075a7783365d9e51f0533c`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 150 | 76.00% | 75.89% | 75.33% | 75.33% | 75.33% |
| capability_substitution | 150 | 0.67% | 0.67% | 0.67% | 0.67% | 0.67% |
| distractor | 150 | 86.67% | 79.71% | 67.33% | 67.33% | 67.33% |
| ephemeral_interface | 150 | 81.33% | 68.39% | 53.33% | 53.33% | 53.33% |
| original | 150 | 94.67% | 87.50% | 74.00% | 74.00% | 74.00% |

## 条件化指标

- Interface Equivariance Accuracy：53.33%
- All-worlds Group Accuracy：0.67%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 58.00% | 87 / 150 |
| original + capability_substitution | 0.67% | 1 / 150 |
| original + distractor | 64.00% | 96 / 150 |
| original + ephemeral_interface | 49.33% | 74 / 150 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
