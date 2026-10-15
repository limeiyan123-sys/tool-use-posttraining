# fctw 单工具能力验证 Probe

## 总体结果

- 样本数：90
- 原协议格式有效率：95.56%
- 原协议 Compatibility Accuracy：66.67%
- 显式协议格式有效率：96.67%
- 显式 Compatibility Accuracy：77.78%
- 显式可解析样本语义准确率：80.46%
- Violation Localization Accuracy：0.00%
- Violation Field Accuracy：20.00%
- Violation Constraint Accuracy：5.00%
- Decision Consistency：56.63%

## 分样本类型

| 类型 | N | Direct Compat | Explicit Compat | Localization | Consistency |
|---|---:|---:|---:|---:|---:|
| functional_mismatch | 30 | 100.00% | 96.67% | 0.00% | 100.00% |
| near_miss | 30 | 13.33% | 96.67% | 0.00% | 14.81% |
| valid | 30 | 86.67% | 40.00% | N/A | 51.85% |

## 分约束族

| 约束族 | N | Direct Compat | Explicit Compat | Localization |
|---|---:|---:|---:|---:|
| enum | 20 | 55.00% | 70.00% | 0.00% |
| functional_mismatch | 30 | 100.00% | 96.67% | 0.00% |
| numeric_range | 20 | 45.00% | 75.00% | 0.00% |
| string_pattern | 20 | 50.00% | 60.00% | 0.00% |

## 解释边界

- 显式协议未参与这些模型的训练，格式有效率与可解析样本语义准确率分开报告。
- Probe 来自已使用过的诊断集，只用于定位瓶颈，不作为新方法的独立最终证据。
- 非法输出不自动修复；仅移除模型生成的固定 `<|im_end|>` 终止标记。
