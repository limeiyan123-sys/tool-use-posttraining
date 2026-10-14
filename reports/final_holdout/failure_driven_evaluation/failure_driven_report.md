# 最终 Holdout 失

> 这是本项目冻结子集上的结果；BFCL-lite 不是 BFCL 官方排行榜成绩。

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：70
- 总耗时：110.53 秒
- 峰值显存：3.076 GiB
- 预测文件 SHA-256：`14f10fd13db0b28530994fc29a09159883b9d32f5a59b3b746629919bd8403b9`

## 分轨指标

| 轨道 | 样本数 | 解析率 | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |
|---|---:|---:|---:|---:|---:|---:|
| bfcl_holdout_v1 | 70 | 100.00% | 100.00% | 93.94% | 81.43% | N/A |

## 分类别指标

| 类别 | 样本数 | Tool Accuracy | Arg Accuracy | Execution Success |
|---|---:|---:|---:|---:|
| bfcl_holdout_v1::irrelevance | 10 | 100.00% | 100.00% | N/A |
| bfcl_holdout_v1::multiple | 20 | 100.00% | 96.25% | N/A |
| bfcl_holdout_v1::parallel | 10 | 100.00% | 88.33% | N/A |
| bfcl_holdout_v1::parallel_multiple | 10 | 100.00% | 90.06% | N/A |
| bfcl_holdout_v1::simple_python | 20 | 100.00% | 93.33% | N/A |

## 口径说明

- Tool Accuracy：预测调用的工具名多重集合与标准答案完全一致。
- Arg Accuracy：工具选择正确后按字段计算；工具选择错误时记为 0。
- Arg Exact：所有参数字段完全正确，允许 BFCL 标注中声明的候选值。
- Execution Success：仅在 controlled_v1 的可执行样本上统计，要求解析、工具、参数和 Schema 全部通过。
- 评测器不修复非法 JSON，不对缺失参数进行猜测，也不静默回退。
