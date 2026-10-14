# fdsrc FCTW 独立评测

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：120
- 总耗时：185.473 秒
- 峰值显存：3.069 GiB
- 预测 SHA-256：`a9c9f41c60d64b10d991ed54c7aff990731e18bec2f8495995e7d20aa78286b4`

## 四象限

| 世界 | 样本数 | Decision Exact | Tool Accuracy | Arg Exact | False Rejection | Wrong Tool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E00 | 30 | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| E01 | 30 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| E10 | 30 | 80.00% | 96.67% | 80.00% | 0.00% | 3.33% |
| E11 | 30 | 63.33% | 73.33% | 63.33% | 0.00% | 26.67% |

## 约束泛化

| 约束族 | 样本数 | Decision Exact | False Rejection | Wrong Tool |
| --- | ---: | ---: | ---: | ---: |
| enum | 40 | 62.50% | 0.00% | 32.50% |
| numeric_range | 40 | 60.00% | 0.00% | 35.00% |
| string_pattern（训练未见） | 40 | 60.00% | 0.00% | 30.00% |

## 关键象限分约束结果

| 约束族 | E01 Decision | E11 Decision | E11 False Rejection |
| --- | ---: | ---: | ---: |
| enum | 0.00% | 70.00% | 0.00% |
| numeric_range | 0.00% | 60.00% | 0.00% |
| string_pattern | 0.00% | 60.00% | 0.00% |

- Overall Decision Exact：60.83%
- E01 Capability Substitution Accuracy：0.00%
- All-worlds Group Accuracy：0.00%

## 评测纪律

- 30 个基础意图均未用于 FCTW 训练或此前诊断与最终评测。
- 同一意图的四个世界按组统计，工具数量保持一致。
- `string_pattern` 为训练阶段未出现的能力约束族。
- 非法输出不自动修复。
