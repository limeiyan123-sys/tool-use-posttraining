# zero_shot 单工具能力验证 Probe

## 总体结果

- 样本数：90
- 原协议格式有效率：94.44%
- 原协议 Compatibility Accuracy：58.89%
- 显式协议格式有效率：98.89%
- 显式 Compatibility Accuracy：76.67%
- 显式可解析样本语义准确率：77.53%
- Violation Localization Accuracy：26.67%
- Violation Field Accuracy：45.00%
- Violation Constraint Accuracy：28.33%
- Decision Consistency：40.48%

## 分样本类型

| 类型 | N | Direct Compat | Explicit Compat | Localization | Consistency |
|---|---:|---:|---:|---:|---:|
| functional_mismatch | 30 | 83.33% | 100.00% | 53.33% | 86.21% |
| near_miss | 30 | 0.00% | 100.00% | 0.00% | 0.00% |
| valid | 30 | 93.33% | 30.00% | N/A | 33.33% |

## 分约束族

| 约束族 | N | Direct Compat | Explicit Compat | Localization |
|---|---:|---:|---:|---:|
| enum | 20 | 50.00% | 65.00% | 0.00% |
| functional_mismatch | 30 | 83.33% | 100.00% | 53.33% |
| numeric_range | 20 | 45.00% | 75.00% | 0.00% |
| string_pattern | 20 | 45.00% | 55.00% | 0.00% |

## 解释边界

- 显式协议未参与这些模型的训练，格式有效率与可解析样本语义准确率分开报告。
- Probe 来自已使用过的诊断集，只用于定位瓶颈，不作为新方法的独立最终证据。
- 非法输出不自动修复；仅移除模型生成的固定 `<|im_end|>` 终止标记。
