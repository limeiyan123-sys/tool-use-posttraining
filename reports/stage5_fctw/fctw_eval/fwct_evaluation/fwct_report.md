# fwct FCTW 独立评测

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：120
- 总耗时：184.204 秒
- 峰值显存：3.069 GiB
- 预测 SHA-256：`c8bbba8a124fc0f31e904686dc14d683eadcd4e9f30cec139089c5da8e3e76f9`

## 四象限

| 世界 | 样本数 | Decision Exact | Tool Accuracy | Arg Exact | False Rejection | Wrong Tool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E00 | 30 | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| E01 | 30 | 13.33% | 13.33% | 13.33% | 0.00% | 86.67% |
| E10 | 30 | 83.33% | 90.00% | 83.33% | 10.00% | 0.00% |
| E11 | 30 | 63.33% | 73.33% | 63.33% | 3.33% | 23.33% |

## 约束泛化

| 约束族 | 样本数 | Decision Exact | False Rejection | Wrong Tool |
| --- | ---: | ---: | ---: | ---: |
| enum | 40 | 67.50% | 2.50% | 27.50% |
| numeric_range | 40 | 65.00% | 2.50% | 32.50% |
| string_pattern（训练未见） | 40 | 62.50% | 5.00% | 22.50% |

## 关键象限分约束结果

| 约束族 | E01 Decision | E11 Decision | E11 False Rejection |
| --- | ---: | ---: | ---: |
| enum | 10.00% | 70.00% | 0.00% |
| numeric_range | 20.00% | 50.00% | 0.00% |
| string_pattern | 10.00% | 70.00% | 10.00% |

- Overall Decision Exact：65.00%
- E01 Capability Substitution Accuracy：13.33%
- All-worlds Group Accuracy：3.33%

## 评测纪律

- 30 个基础意图均未用于 FCTW 训练或此前诊断与最终评测。
- 同一意图的四个世界按组统计，工具数量保持一致。
- `string_pattern` 为训练阶段未出现的能力约束族。
- 非法输出不自动修复。
