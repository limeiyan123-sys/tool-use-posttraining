# E2 v3 分层语义抽查

## 抽查方法

按 `constraint_family × category` 分层，每层按冻结文件顺序抽取前两对，共 14 对。逐项核对：

1. 用户请求是否明确给出目标字段值；
2. E+ 是否允许全部目标值；
3. E- 是否排除至少一个完成请求所必需的目标值，且不存在同合同内替代工具；
4. 两侧是否只改变决定可满足性的约束值，而没有“是否被编辑”的标签痕迹；
5. 是否存在默认值、严格边界或业务语义歧义。

## 抽查记录

| Pair | 约束 | 类别 | 目标字段 | Gold | 结论 |
|---|---|---|---|---|---|
| `multiple_106` | enum | multiple | distance | 4 | 通过 |
| `multiple_30` | enum | multiple | width | 5 | 通过 |
| `parallel_73` | enum | parallel | value | 5, 3 | 通过 |
| `parallel_5` | enum | parallel | streaming_service | Netflix, Hulu | 通过 |
| `parallel_multiple_43` | enum | parallel_multiple | museum_location | Philadelphia, New York | 通过 |
| `parallel_multiple_13` | enum | parallel_multiple | final_temperature | 100, 100 | 通过 |
| `simple_python_4` | enum | simple_python | c | 5 | 通过 |
| `simple_python_358` | enum | simple_python | dish | cookie | 通过 |
| `multiple_82` | maximum | multiple | tempo | 120 | 通过 |
| `multiple_152` | maximum | multiple | start_year | 1871 | 通过 |
| `parallel_96` | maximum | parallel | charge1 | 5, 5 | 通过 |
| `parallel_67` | maximum | parallel | radius | 5, 10 | 通过 |
| `simple_python_146` | maximum | simple_python | days | 3 | 通过 |
| `simple_python_158` | maximum | simple_python | from_year | 2012 | 通过 |

结果：14/14 通过本 Pilot 的闭世界语义标准。样本量不足以证明总体一致率达到 95%，因此这里只报告“抽查未再发现歧义”，不外推为完整人工标签准确率。

## 修订来源

v2 抽查曾发现 `multiple_186` 的 `under 30 minutes` 与 `maximum=30` 存在开闭区间歧义。v3 新增严格上界措辞检查，共排除 2 对：`multiple_186` 与 `multiple_89`。失败记录保留在 `data_v2/REVIEW_STATUS.md`。
