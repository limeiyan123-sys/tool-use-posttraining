# fdsrc 单工具能力验证 Probe

## 总体结果

- 样本数：90
- 原协议格式有效率：95.56%
- 原协议 Compatibility Accuracy：64.44%
- 显式协议格式有效率：97.78%
- 显式 Compatibility Accuracy：77.78%
- 显式可解析样本语义准确率：79.55%
- Violation Localization Accuracy：0.00%
- Violation Field Accuracy：20.00%
- Violation Constraint Accuracy：0.00%
- Decision Consistency：47.62%

## 分样本类型

| 类型 | N | Direct Compat | Explicit Compat | Localization | Consistency |
|---|---:|---:|---:|---:|---:|
| functional_mismatch | 30 | 100.00% | 93.33% | 0.00% | 100.00% |
| near_miss | 30 | 0.00% | 100.00% | 0.00% | 0.00% |
| valid | 30 | 93.33% | 40.00% | N/A | 42.86% |

## 分约束族

| 约束族 | N | Direct Compat | Explicit Compat | Localization |
|---|---:|---:|---:|---:|
| enum | 20 | 50.00% | 65.00% | 0.00% |
| functional_mismatch | 30 | 100.00% | 93.33% | 0.00% |
| numeric_range | 20 | 45.00% | 75.00% | 0.00% |
| string_pattern | 20 | 45.00% | 70.00% | 0.00% |

## 解释边界

- 显式协议未参与这些模型的训练，格式有效率与可解析样本语义准确率分开报告。
- Probe 来自已使用过的诊断集，只用于定位瓶颈，不作为新方法的独立最终证据。
- 非法输出不自动修复；仅移除模型生成的固定 `<|im_end|>` 终止标记。
