# Stage 7：Tool-Use SFT 是否形成历史工具能力依赖？

## 1. 研究问题

本阶段检验以下假设：Tool-Use SFT 是否会使小参数模型记住训练时的“工具名称—能力”对应关系，从而在运行时能力说明发生变化后，仍然选择过去正确、当前错误的工具。

本实验只做诊断与一次最小受控 LoRA，不设计新算法。所有指标来自真实模型输出，测试 Gold 与能力契约由确定性规则生成。

## 2. 实验设置

### 2.1 模型

- 基础模型：`Qwen2.5-1.5B-Instruct`。
- 普通 Tool-Use LoRA：历史 `stage1_lora_baseline` adapter。
- FDSRC：历史 `stage3_failure_driven` adapter。
- 受控历史映射 SFT：从同一基础模型训练的新 LoRA，明确只暴露 Original Environment。

### 2.2 配对工具环境

构建 6 个工具家族：报表导出、图片转换、数据打包、文档翻译、账单币种换算、图表渲染。每个家族包含 4 个独立测试意图，共 24 个基础任务、48 个环境样本。

每个配对满足：

- Original 与 Updated 使用完全相同的用户请求。
- 工具名称、数量、参数 Schema、工具顺序与提示格式保持一致。
- 两个工具都能以同一结构合法调用。
- Updated 只交换两个中性名称工具的能力描述与 Mock Tool 行为。
- Original 正确工具位于 alpha/beta 的任务各 12 个，位置标签平衡。
- 测试请求与受控 SFT 请求完全独立。

测试集 SHA-256：`f32d293d69db13e24dac907642f0140c3344dd236bd27ad4f33de58faa693a75`。

### 2.3 受控 LoRA

| 配置 | 数值 |
|---|---:|
| Train / Dev | 48 / 12 |
| Epoch | 4 |
| Optimizer steps | 24 |
| Learning rate | 1e-4 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target | all-linear |
| Trainable parameters | 9,232,384（0.5945%） |
| Effective batch size | 8 |
| Supervised tokens per sample | 22 |
| Train runtime | 43.65 秒 |
| Train loss | 0.10893 |
| Final dev loss | 0.000686 |

只监督 assistant 的结构化工具调用，system、工具描述和用户请求全部被 `-100` 掩码。训练显存被限制为 GPU 0 总显存的 20%，未使用 GPU 1。

adapter SHA-256：`358c045252db929800fdd2d99ae95f5edd85806da680635b79922a370447be435`。392 个 adapter 张量均为非零。

### 2.4 推理与评测

- 固定贪心解码，`do_sample=false`，`max_new_tokens=96`。
- 严格解析 `<tool_call>`，不进行宽松修复。
- Mock Tool 根据冻结能力契约检查工具能力与参数。
- Bootstrap 以 24 个基础任务为聚类单位进行 2,000 次配对重采样。

## 3. 主要结果

| 模型 | Original Accuracy | Updated Accuracy | Pair Success | Stale Choice（全部 Updated） | Semantic Error | False Reject |
|---|---:|---:|---:|---:|---:|---:|
| Base | 100.00% | 100.00% | 100.00% | 0/24（0.00%） | 0.00% | 0.00% |
| 普通 LoRA | 100.00% | 100.00% | 100.00% | 0/24（0.00%） | 0.00% | 0.00% |
| FDSRC | 100.00% | 100.00% | 100.00% | 0/24（0.00%） | 0.00% | 0.00% |
| 受控历史映射 SFT | 100.00% | 100.00% | 100.00% | 0/24（0.00%） | 0.00% | 0.00% |

四个模型在六个工具家族中的 Original 与 Updated Accuracy 均为 100%。由于所有配对均正确，各比例的经验 Bootstrap 区间退化为单点；这不代表总体真实误差率必然为零，只表示当前 24 个配对中没有观察到失败。

## 4. 结果能够证明什么

1. 当前配对任务对基础模型是可解的，工具名称和位置没有造成明显偏置。
2. 普通 Tool-Use LoRA 与 FDSRC 没有损害模型服从当前能力描述的能力。
3. 更关键的是，受控 SFT 明确反复暴露固定的 Original 工具—能力映射，训练 loss 与非零 adapter 权重证明优化实际发生；模型在 Updated Environment 仍全部选择当前正确工具。
4. 因此，在“中性工具名、明确能力描述、两个合法候选、单步调用”的受控条件下，H1 没有得到支持。当前工具契约足以覆盖训练形成的名称—能力对应关系。
5. Stage 6 观察到的动态环境失败不能被简单归因于“模型记住了旧工具名”。更可能的瓶颈仍是 Near-miss 约束判断、用户需求到 Schema 字段的绑定，以及兼容性证据进入最终决策的方式。

## 5. 不能证明什么

- 不能证明 Tool-Use SFT 在所有条件下都不会产生历史依赖。
- 当前能力描述短且直接；长工具列表、模糊描述、上下文冲突或更强训练暴露可能得到不同结果。
- 受控 LoRA 规模较小且只有一个随机种子，结论仅适用于本次诊断范围。
- 基础模型已经达到满分，因此无法衡量 SFT 是否改善 Original，只能检验它是否造成额外退化。
- 历史普通 LoRA 与 FDSRC 由较新 PEFT 版本保存，在当前环境加载时忽略了与核心 LoRA 无关的新配置字段；其结果作为补充证据，不承担因果识别。

## 6. 为什么停止扩展实验

按照预先设定的情况 C，受控 SFT 没有损害 Updated Environment。当前没有错误样本可供位置交换、简化描述或 Oracle 兼容性诊断，因此继续运行这些检查不能区分竞争解释。也没有证据支持投入 7B、更多种子或更长训练。

## 7. 阶段性研究判断

本阶段的负面结果具有明确研究价值：**在清晰的当前工具契约下，小参数模型并未表现出由 Tool-Use SFT 引起的旧能力选择惯性。**

下一阶段不应继续围绕工具名称记忆设计复杂算法。更值得研究的是 Stage 6 已定位的 Requirement-to-Schema Binding：当能力限制隐含在参数 enum、范围或可选字段语义中时，模型为何不能稳定把用户要求与当前 Schema 对齐。新的独立评测应降低表面描述的直接性，但仍保持 Gold 可验证，避免为了制造退化而引入歧义。

## 8. 复现与产物

- 数据与能力契约：`data/stage7/historical_dependency_v1/`
- 数据构造、推理、评测：`scripts/stage7_historical_dependency.py`
- 受控训练：`scripts/train_stage7_controlled_sft.py`
- 新 LoRA：`checkpoints/stage7_controlled_history_sft/`
- 逐样本预测与指标：`reports/stage7_historical_dependency/`
