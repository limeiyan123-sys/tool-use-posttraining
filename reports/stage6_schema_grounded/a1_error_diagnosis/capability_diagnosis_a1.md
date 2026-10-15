# Stage 6 动态工具调用分层错误诊断

## 指标定义

- Call/Reject：严格解析成功，且是否调用与 Gold 一致。
- Tool Selection：以全部应调用样本为分母，错误拒绝和非法输出计错。
- Argument Exact：仅在严格解析且工具选择正确的样本上计算条件准确率。
- End-to-end：沿用既有 Decision Exact，不用条件指标替代最终表现。

## 核心结果

| 方法 | 场景 | N | Call/Reject | Tool@ShouldCall | ArgExact@CorrectTool | End-to-end |
|---|---|---:|---:|---:|---:|---:|
| FDSRC | overall | 300 | 82.33% | 81.77% | 68.07% | 52.67% |
| FDSRC | capability_substitution | 60 | 23.33% | 0.00% | N/A | 0.00% |
| FDSRC | distractor | 60 | 98.33% | 89.83% | 67.92% | 60.00% |
| FDSRC | original | 60 | 100.00% | 95.00% | 71.93% | 68.33% |
| FWCT | overall | 300 | 87.67% | 76.96% | 69.43% | 62.33% |
| FWCT | capability_substitution | 60 | 78.33% | 0.00% | N/A | 56.67% |
| FWCT | distractor | 60 | 76.67% | 71.67% | 74.42% | 53.33% |
| FWCT | original | 60 | 98.33% | 95.00% | 73.68% | 70.00% |
| FCTW | overall | 300 | 82.67% | 83.66% | 68.64% | 54.00% |
| FCTW | capability_substitution | 60 | 26.67% | 0.00% | N/A | 3.33% |
| FCTW | distractor | 60 | 98.33% | 94.92% | 71.43% | 66.67% |
| FCTW | original | 60 | 100.00% | 96.67% | 70.69% | 68.33% |

## Substitution 错误计数

| 方法 | Correct | False Call | False Reject | Wrong Tool | Arg Failure | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| FDSRC | 0 | 46 | 0 | 14 | 0 | 0 |
| FWCT | 34 | 12 | 1 | 13 | 0 | 0 |
| FCTW | 2 | 44 | 0 | 14 | 0 | 0 |

## Distractor 错误计数

| 方法 | Correct | False Call | False Reject | Wrong Tool | Arg Failure | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| FDSRC | 36 | 0 | 0 | 6 | 17 | 1 |
| FWCT | 32 | 0 | 14 | 3 | 11 | 0 |
| FCTW | 40 | 0 | 0 | 3 | 16 | 1 |

## 研究判断

1. FCTW 的 Substitution 仅 2/60 端到端正确，主要错误是 44 次 False Call；另有 14 个仍需保留部分合法调用的样本发生 Wrong Tool。参数生成不是该场景的首要瓶颈。
2. FWCT 将 Substitution 的正确拒绝显著提高到 34/60，但仍有 12 次 False Call、13 次 Wrong Tool，说明它学到了一部分能力约束信号，但不稳定。
3. FWCT 的 Distractor 退化直接来自 14 次 False Reject；FCTW 消除了这批错误拒绝，却几乎完全丢失 FWCT 的 Substitution 收益。
4. FCTW 与 FDSRC 在 Original 和 Distractor 的分层指标接近，说明核心冲突位于“语义相关但能力不兼容”的判断，而非一般工具调用格式或普通参数生成。
5. 下一步应使用单工具 Probe 区分：模型是不会判断单个 Schema 的兼容性，还是能够判断但无法把判断整合进多工具行动。
