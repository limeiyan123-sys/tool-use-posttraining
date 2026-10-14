# Stage 1 冻结评测协议

## 目标

本协议用于比较 `Qwen2.5-1.5B-Instruct` 的 Zero-shot、普通 LoRA SFT 和后续改进方法。测试集在任何正式训练前冻结，后续方法必须复用相同文件与评分代码。

## 双轨设计

### controlled_v1

本项目构造的 28 条受控样本，包含：

- `simple_selection`：单工具基础调用；
- `tool_competition`：相似工具竞争；
- `argument_constraints`：数组、布尔值、枚举、嵌套对象等参数约束；
- `unseen_schema`：使用训练阶段禁止出现的工具名与参数名；
- `irrelevance`：没有相关工具时应拒绝调用。

前 24 条工具调用样本支持本地确定性执行判定。`unseen_schema` 的名称及精确 Schema 必须加入训练数据禁用清单。

### bfcl_lite_v1

从 BFCL V4 的 `simple_python`、`multiple`、`parallel`、`parallel_multiple` 和 `irrelevance` 五个单轮类别中，按固定种子和样本 ID 哈希选择 70 条。数据源固定到 Gorilla Git 提交：

`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`

该子集只用于低成本外部有效性检查，不是完整 BFCL，也不能作为官方排行榜成绩。公开 BFCL 数据可能出现在基础模型预训练语料中，因此不能将其称为严格未见测试。

## 指标

| 指标 | 定义 |
|---|---|
| Parse Rate | 输出能否在不修复的前提下解析为严格 `<tool_call>` JSON |
| Tool Accuracy | 预测工具名多重集合与标准答案完全一致 |
| Argument Accuracy | 工具选择正确后按参数字段计分，工具选择错误时记 0 |
| Argument Exact Match | 所有参数字段精确正确 |
| Execution Success Rate | 仅受控可执行样本统计；解析、工具、参数、Schema 均通过才记成功 |

并行调用按工具名做最优对齐，不依赖调用顺序。数值允许整数与浮点数的等值表示，布尔值不允许与整数混同。BFCL 标注候选值包含空字符串时，同时接受省略字段或显式空字符串。除 BFCL 标注明确给出的候选值外，不进行同义改写、日期猜测或 JSON 修复。

## 生成设置

- 确定性解码：`do_sample=false`；
- `max_new_tokens=256`；
- 固定种子：`20261014`；
- 使用模型原生 Qwen tool template；
- 每种方法只生成一次正式预测文件，预测文件和数据文件均记录 SHA-256。

## 污染控制

1. 在选择和清洗 SFT 数据时，对冻结测试集的用户文本、工具名、描述和 Schema 做精确及近似去重。
2. `controlled_v1::unseen_schema` 中的工具名和参数名不得进入训练集。
3. 不根据测试集错误手工加入同题或同 Schema 训练样本；困难样本只能从训练/开发集失败中构造。
4. 如发现测试标注错误，不修改 `frozen_v1`，而是建立新版本并同时报告旧版本结果。

## 错误纪律

评测器禁止静默 fallback。非法 JSON、缺失参数、额外参数、错误类型和未声明工具均保留原始输出并显式归类。
