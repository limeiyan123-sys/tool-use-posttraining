# Stage 3 Schema-only 消融结果

> 这是本项目冻结子集上的结果；BFCL-lite 不是 BFCL 官方排行榜成绩。

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：98
- 总耗时：131.54 秒
- 峰值显存：3.067 GiB
- 预测文件 SHA-256：`df4238808dae7cf7e7c886af3d8a58dd9aafb9b4fddf576387753dba56ddb8a2`

## 分轨指标

| 轨道 | 样本数 | 解析率 | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |
|---|---:|---:|---:|---:|---:|---:|
| bfcl_lite_v1 | 70 | 98.57% | 91.43% | 86.87% | 77.14% | N/A |
| controlled_v1 | 28 | 100.00% | 100.00% | 93.75% | 85.71% | 83.33% |

## 分类别指标

| 类别 | 样本数 | Tool Accuracy | Arg Accuracy | Execution Success |
|---|---:|---:|---:|---:|
| bfcl_lite_v1::irrelevance | 10 | 90.00% | 90.00% | N/A |
| bfcl_lite_v1::multiple | 20 | 90.00% | 86.25% | N/A |
| bfcl_lite_v1::parallel | 10 | 80.00% | 80.00% | N/A |
| bfcl_lite_v1::parallel_multiple | 10 | 90.00% | 83.06% | N/A |
| bfcl_lite_v1::simple_python | 20 | 100.00% | 91.25% | N/A |
| controlled_v1::argument_constraints | 6 | 100.00% | 95.83% | 83.33% |
| controlled_v1::irrelevance | 4 | 100.00% | 100.00% | N/A |
| controlled_v1::simple_selection | 6 | 100.00% | 100.00% | 100.00% |
| controlled_v1::tool_competition | 6 | 100.00% | 80.56% | 66.67% |
| controlled_v1::unseen_schema | 6 | 100.00% | 94.44% | 83.33% |

## 口径说明

- Tool Accuracy：预测调用的工具名多重集合与标准答案完全一致。
- Arg Accuracy：工具选择正确后按字段计算；工具选择错误时记为 0。
- Arg Exact：所有参数字段完全正确，允许 BFCL 标注中声明的候选值。
- Execution Success：仅在 controlled_v1 的可执行样本上统计，要求解析、工具、参数和 Schema 全部通过。
- 评测器不修复非法 JSON，不对缺失参数进行猜测，也不静默回退。
