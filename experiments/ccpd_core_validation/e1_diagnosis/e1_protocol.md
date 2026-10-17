# E1 冻结实验协议

## 研究问题

本实验区分两种行为变化：

1. 模型整体更倾向于调用或拒绝；
2. 模型对同一请求下可满足合同与不可满足合同的排序能力发生变化。

E1 不训练模型。Base、FDSRC、FWCT、FCTW 使用完全相同的样本、提示和评分方法。

## 数据冻结规则

- 来源：`capability_probe_v1` 中同一 `base_id` 的 `valid` 与 `near_miss`。
- 主分析：被施加能力约束的目标字段必须是必填字段。
- 敏感性分析：目标字段为可选字段的样本单独报告，不与主结论混合。
- 不使用 `functional_mismatch`，因为它不能隔离细粒度合同可满足性。
- 每对样本必须具有完全相同的用户请求，且两侧均只有一个候选工具。
- E+ 的 Gold 调用来自冻结的 FCTW E10；E- 的约束必须排除该 Gold 目标值。

## 协议 A：原生工具调用

直接复用 Stage 6 的确定性生成结果。严格解析 `<tool_call>`：

- 至少一个合法调用：CALL；
- 合法且无调用：DECLINE；
- 无法解析：INVALID，不能自动修复为拒绝。

## 协议 B：双顺序受控决策探针

把完整用户请求与唯一工具合同作为普通文本输入，要求只在 A/B 中选择。运行两个标签映射：

- 顺序 1：A=CALL，B=DECLINE；
- 顺序 2：A=DECLINE，B=CALL。

两个标签必须在当前上下文中各自恰好对应一个 Token，否则实验直接失败。定义：

```text
s1 = logit(A) - logit(B)
s2 = logit(B) - logit(A)
s  = (s1 + s2) / 2
```

`s` 仅是该受控二选一任务的相对分数，不解释为模型原生工具调用概率，也不等同于模型内部真实存在的独立偏置项。

## 冻结指标

- Pair Success：E+ 选择 CALL 且 E- 选择 DECLINE。
- Paired Concordance：`s(E+) > s(E-)`；平分单独报告。
- Valid Call Rate：E+ 的 CALL 比例。
- Invalid Call Rate：E- 的 CALL 比例。
- False Reject Rate：E+ 的非 CALL 比例；原生协议中 INVALID 单独列出。
- Pair Score Difference：`s(E+) - s(E-)`。
- Mean Decision Score：`(s(E+) + s(E-)) / 2`，仅作为整体调用倾向的行为代理。
- Native/Probe Agreement：协议 A 与协议 B 的逐侧决策一致率。

不确定性以基础请求为单位执行配对 Bootstrap，固定种子 `20261017`，重复 `10000` 次。

## 预先规定的解释规则

- 若 FWCT 主要改变 Mean Decision Score，而 Paired Concordance、Pair Score Difference 和 Pair Success 没有稳定改善，支持“整体调用倾向移动”。
- 若 FWCT 稳定改善后三项，支持“合同判别改善”。
- 若两类指标同时变化，结论为混合机制。
- 若协议 A 与 B 一致率较低，必须把“输出协议差异”保留为竞争解释，不能只用探针支持 CCPD。
