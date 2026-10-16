# Stage 9 参数语义绑定因子实验结果

| 模型/策略 | A Decision | B Decision | C Decision | D Decision | A Binding | B Binding | C Binding | D Binding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_explicit_mapping | 未运行 | 95.83% | 未运行 | 25.00% | 未运行 | 100.00% | 未运行 | 48.81% |
| base_description_alignment | 未运行 | 91.67% | 未运行 | 54.17% | 未运行 | 97.62% | 未运行 | 95.24% |

## 因子效应

负值表示相对 A 退化；交互项为 D - C - B + A。

| 模型/策略 | 标识符效应 | 顺序效应 | Decision 交互 | Binding 交互 |
|---|---:|---:|---:|---:|

## 解释边界

- 24 个基础任务用于机制诊断，区间按基础任务聚类重采样，不把四个派生条件视为独立样本。
- description_alignment 是固定描述抽取规则的 PA-Tool-inspired 简化对照，不等同于 PA-Tool 原方法。
- explicit_mapping 向模型提供了额外标识符映射，只作为信息可达性的诊断上界。
