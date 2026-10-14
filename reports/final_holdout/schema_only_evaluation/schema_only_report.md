# 最终 Holdout Schema-only 结果

> 这是本项目冻结子集上的结果；BFCL-lite 不是 BFCL 官方排行榜成绩。

## 运行信息

- 模型：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`
- 样本数：70
- 总耗时：110.976 秒
- 峰值显存：3.076 GiB
- 预测文件 SHA-256：`3a0e9de32bed2071562f1e4a489f9ac598dedf5a02de307ed29acedd3d695e94`

## 分轨指标

| 轨道 | 样本数 | 解析率 | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |
|---|---:|---:|---:|---:|---:|---:|
| bfcl_holdout_v1 | 70 | 100.00% | 98.57% | 91.56% | 77.14% | N/A |

## 分类别指标

| 类别 | 样本数 | Tool Accuracy | Arg Accuracy | Execution Success |
|---|---:|---:|---:|---:|
| bfcl_holdout_v1::irrelevance | 10 | 100.00% | 100.00% | N/A |
| bfcl_holdout_v1::multiple | 20 | 100.00% | 91.25% | N/A |
| bfcl_holdout_v1::parallel | 10 | 100.00% | 88.33% | N/A |
| bfcl_holdout_v1::parallel_multiple | 10 | 100.00% | 91.72% | N/A |
| bfcl_holdout_v1::simple_python | 20 | 95.00% | 89.17% | N/A |

## 口径说明

- Tool Accuracy：预测调用的工具名多重集合与标准答案完全一致。
- Arg Accuracy：工具选择正确后按字段计算；工具选择错误时记为 0。
- Arg Exact：所有参数字段完全正确，允许 BFCL 标注中声明的候选值。
- Execution Success：仅在 controlled_v1 的可执行样本上统计，要求解析、工具、参数和 Schema 全部通过。
- 评测器不修复非法 JSON，不对缺失参数进行猜测，也不静默回退。
