# Stage 1 Zero-shot Baseline 结果

> 这是本项目冻结子集上的结果；BFCL-lite 不是 BFCL 官方排行榜成绩。

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：98
- 总耗时：69.878 秒
- 峰值显存：2.955 GiB
- 预测文件 SHA-256：`19b4538ff0f5ca10b947847d9aab39725558e8d2483302d11a87e1ac819aea4e`

## 分轨指标

| 轨道 | 样本数 | 解析率 | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |
|---|---:|---:|---:|---:|---:|---:|
| bfcl_lite_v1 | 70 | 98.57% | 90.00% | 85.12% | 74.29% | N/A |
| controlled_v1 | 28 | 100.00% | 96.43% | 90.18% | 78.57% | 79.17% |

## 分类别指标

| 类别 | 样本数 | Tool Accuracy | Arg Accuracy | Execution Success |
|---|---:|---:|---:|---:|
| bfcl_lite_v1::irrelevance | 10 | 80.00% | 80.00% | N/A |
| bfcl_lite_v1::multiple | 20 | 95.00% | 87.92% | N/A |
| bfcl_lite_v1::parallel | 10 | 80.00% | 80.00% | N/A |
| bfcl_lite_v1::parallel_multiple | 10 | 80.00% | 72.50% | N/A |
| bfcl_lite_v1::simple_python | 20 | 100.00% | 93.75% | N/A |
| controlled_v1::argument_constraints | 6 | 100.00% | 90.28% | 66.67% |
| controlled_v1::irrelevance | 4 | 75.00% | 75.00% | N/A |
| controlled_v1::simple_selection | 6 | 100.00% | 100.00% | 100.00% |
| controlled_v1::tool_competition | 6 | 100.00% | 86.11% | 66.67% |
| controlled_v1::unseen_schema | 6 | 100.00% | 94.44% | 83.33% |

## 口径说明

- Tool Accuracy：预测调用的工具名多重集合与标准答案完全一致。
- Arg Accuracy：工具选择正确后按字段计算；工具选择错误时记为 0。
- Arg Exact：所有参数字段完全正确，允许 BFCL 标注中声明的候选值。
- Execution Success：仅在 controlled_v1 的可执行样本上统计，要求解析、工具、参数和 Schema 全部通过。
- 评测器不修复非法 JSON，不对缺失参数进行猜测，也不静默回退。
