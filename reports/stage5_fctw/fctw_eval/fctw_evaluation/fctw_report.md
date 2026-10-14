# fctw FCTW 独立评测

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：120
- 总耗时：165.135 秒
- 峰值显存：3.069 GiB
- 预测 SHA-256：`37b848308c16901923ad5ecaf16c0ee55f60c0d367400ea9b7c73f4a39c8ce06`

## 四象限

| 世界 | 样本数 | Decision Exact | Tool Accuracy | Arg Exact | False Rejection | Wrong Tool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E00 | 30 | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| E01 | 30 | 6.67% | 6.67% | 6.67% | 0.00% | 93.33% |
| E10 | 30 | 86.67% | 93.33% | 86.67% | 6.67% | 0.00% |
| E11 | 30 | 70.00% | 80.00% | 70.00% | 0.00% | 20.00% |

## 约束泛化

| 约束族 | 样本数 | Decision Exact | False Rejection | Wrong Tool |
| --- | ---: | ---: | ---: | ---: |
| enum | 40 | 65.00% | 2.50% | 30.00% |
| numeric_range | 40 | 65.00% | 2.50% | 32.50% |
| string_pattern（训练未见） | 40 | 67.50% | 0.00% | 22.50% |

- Overall Decision Exact：65.83%
- E01 Capability Substitution Accuracy：6.67%
- All-worlds Group Accuracy：3.33%

## 评测纪律

- 30 个基础意图均未用于 FCTW 训练或此前诊断与最终评测。
- 同一意图的四个世界按组统计，工具数量保持一致。
- `string_pattern` 为训练阶段未出现的能力约束族。
- 非法输出不自动修复。
