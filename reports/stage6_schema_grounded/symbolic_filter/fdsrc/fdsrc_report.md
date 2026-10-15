# fdsrc Symbolic Schema Filter

## Decision Exact

| 场景 | 原始 | 过滤后 | 变化 |
|---|---:|---:|---:|
| overall | 52.67% | 67.33% | +14.67 pp |
| capability_substitution | 0.00% | 66.67% | +66.67 pp |
| distractor | 60.00% | 60.00% | +0.00 pp |
| original | 68.33% | 68.33% | +0.00 pp |

## 过滤统计

- 被过滤调用数：86
- 受影响样本数：62
- 错误转正确：44
- 正确转错误：0
- 原因：{"schema_violation": 78, "unknown_tool": 8}

## 解释边界

- 该基线只验证模型已经生成的参数是否满足当前 JSON Schema，不生成或猜测缺失需求。
- 用户要求的值若对应 Schema 中的可选字段，而模型省略该字段，纯 Schema 校验无法发现。
- 过滤后不自动重试模型，因此它衡量确定性拒绝能够直接覆盖多少错误，而不是完整 Agent 上限。
- 非法输出保持原样，不使用宽松解析或静默修复。
