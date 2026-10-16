# Stage 9 参数语义绑定因子实验结果

| 模型/策略 | A Decision | B Decision | C Decision | D Decision | A Binding | B Binding | C Binding | D Binding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fdsrc | 75.00% | 75.00% | 66.67% | 54.17% | 92.86% | 92.86% | 90.48% | 84.52% |

## 因子效应

负值表示相对 A 退化；交互项为 D - C - B + A。

| 模型/策略 | 标识符效应 | 顺序效应 | Decision 交互 | Binding 交互 |
|---|---:|---:|---:|---:|
| fdsrc | +0.00 pp | -8.33 pp | -12.50 pp | -5.95 pp |

## 解释边界

- 24 个基础任务用于机制诊断，区间按基础任务聚类重采样，不把四个派生条件视为独立样本。
- description_alignment 是固定描述抽取规则的 PA-Tool-inspired 简化对照，不等同于 PA-Tool 原方法。
- explicit_mapping 向模型提供了额外标识符映射，只作为信息可达性的诊断上界。
