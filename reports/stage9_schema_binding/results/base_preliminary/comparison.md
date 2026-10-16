# Stage 9 参数语义绑定因子实验结果

| 模型/策略 | A Decision | B Decision | C Decision | D Decision | A Binding | B Binding | C Binding | D Binding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 79.17% | 91.67% | 41.67% | 45.83% | 94.05% | 98.81% | 91.67% | 83.33% |

## 因子效应

负值表示相对 A 退化；交互项为 D - C - B + A。

| 模型/策略 | 标识符效应 | 顺序效应 | Decision 交互 | Binding 交互 |
|---|---:|---:|---:|---:|
| base | +12.50 pp | -37.50 pp | -8.33 pp | -13.10 pp |

## 解释边界

- 24 个基础任务用于机制诊断，区间按基础任务聚类重采样，不把四个派生条件视为独立样本。
- description_alignment 是固定描述抽取规则的 PA-Tool-inspired 简化对照，不等同于 PA-Tool 原方法。
- explicit_mapping 向模型提供了额外标识符映射，只作为信息可达性的诊断上界。
