# Stage 1 SFT 数据质量报告

## 数据来源

- 数据集：`Team-ACE/ToolACE`
- 许可证：Apache-2.0
- 官方文件 SHA-256：`ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e`
- 下载方式：Hugging Face 镜像；哈希已与官方 Git LFS 对象核对一致

## 处理结果

| 项目 | 数量 |
|---|---:|
| 原始对话 | 11300 |
| 通过质量与污染检查的工具调用对话 | 8441 |
| 通过质量与污染检查的 no-tool 对话 | 1502 |
| 入选工具调用对话 | 1200 |
| 入选 no-tool 对话 | 300 |
| 训练集 | 1350 |
| 开发集 | 150 |

## 剔除统计

```json
{
  "exact_duplicate": 3,
  "invalid::JSONDecodeError": 288,
  "invalid::SyntaxError": 91,
  "invalid::TypeError": 641,
  "invalid::ValueError": 330,
  "tool_name_overlap": 4
}
```

## 质量策略

- 工具 Schema 必须是可验证 JSON Schema 子集；损坏结构不修复。
- 函数调用只接受已声明工具、关键字参数和 Python 字面量。
- 必填参数缺失、额外参数、角色顺序异常均剔除。
- 与冻结测试集重合的工具名、完整工具定义、精确问题和高相似问题均剔除。
- 训练/开发集按稳定内容哈希划分，不依赖源文件顺序。

## 限制

- ToolACE 是合成数据，领域和表达分布不等同于真实用户流量。
- 本基线采用有限子集控制单卡训练成本，不代表全量 ToolACE 训练效果。
- 普通 LoRA 数据未专门增强 Schema 重命名或困难负例；这些留给 Proposed Method。
