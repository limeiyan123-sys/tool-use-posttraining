# fctw 工具世界诊断结果

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：300
- 总耗时：605.997 秒
- 峰值显存：3.123 GiB
- 预测 SHA-256：`43db72e91583c026fe4a47c1e05240cdd1a6af744f5b92f5d5b28123d6001848`

## 分工具世界结果

| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |
|---|---:|---:|---:|---:|---:|---:|
| capability_removal | 60 | 83.33% | 82.64% | 80.00% | 80.00% | 80.00% |
| capability_substitution | 60 | 3.33% | 3.33% | 3.33% | 3.33% | 3.33% |
| distractor | 60 | 93.33% | 84.27% | 66.67% | 66.67% | 66.67% |
| ephemeral_interface | 60 | 83.33% | 72.78% | 53.33% | 51.67% | 53.33% |
| original | 60 | 96.67% | 87.56% | 68.33% | 68.33% | 68.33% |

## 条件化指标

- Interface Equivariance Accuracy：53.33%
- All-worlds Group Accuracy：1.67%

| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |
|---|---:|---:|
| original + capability_removal | 61.67% | 37 / 60 |
| original + capability_substitution | 3.33% | 2 / 60 |
| original + distractor | 61.67% | 37 / 60 |
| original + ephemeral_interface | 50.00% | 30 / 60 |

## 评测纪律

- 非法输出不自动修复。
- 能力移除和替换后的 Gold 已按当前工具世界重新计算。
- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。
- 本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。
