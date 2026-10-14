# Stage 1 基线实验记录

## 实验结论

在同一组 98 条冻结样本上，普通 LoRA SFT 相比 Zero-shot 呈现一致但有限的改善趋势：Tool Accuracy 提升 2.04 个百分点，Arg Accuracy 提升 1.84 个百分点，参数完全匹配率提升 4.08 个百分点。三项差值的 95% 配对 bootstrap 置信区间均跨过 0，因此本阶段不宣称统计显著改进。

LoRA 的主要收益来自输出格式稳定性、无关请求拒答、常见参数规范化和部分缺失参数补全；主要风险是多工具过调用、必填字段遗漏和未见 Schema 上的拒绝调用。这些现象将直接约束 Stage 2 的错误分析和 Stage 3 的方法设计。

## 公共设置

- 模型：`Qwen2.5-1.5B-Instruct`
- 冻结测试集：`controlled_v1` 28 条，`bfcl_lite_v1` 70 条，共 98 条
- 生成方式：确定性生成，`do_sample=false`，`max_new_tokens=256`
- 评测随机种子：`20261014`
- 评测指标：解析率、Tool Accuracy、Arg Accuracy、Arg Exact、Execution Success
- 统计方法：逐样本配对比较，10,000 次配对 bootstrap
- 重要边界：`bfcl_lite_v1` 是固定抽样的研究子集，不是 BFCL 官方排行榜成绩

## 实验一：Zero-shot

```makefile
experiment_name: stage1_zero_shot
model: Qwen2.5-1.5B-Instruct
dataset: controlled_v1(28) + bfcl_lite_v1(70)
method: zero-shot deterministic tool calling
hyperparameter: do_sample=false, max_new_tokens=256, seed=20261014
result: Tool=91.84%, Arg=86.90%, Arg Exact=76.53%, controlled execution=79.17%
analysis: 基础模型已有较强单工具能力；并行多工具、无关请求拒答、参数单位与必填字段仍是主要错误来源
next_step: 使用经过污染检查的 ToolACE 子集进行普通 LoRA SFT
```

运行信息：

- 推理耗时：69.878 秒
- 峰值显存：2.955 GiB
- 预测 SHA-256：`19b4538ff0f5ca10b947847d9aab39725558e8d2483302d11a87e1acc819aea4e`

## 实验二：LoRA 冒烟实验

```makefile
experiment_name: stage1_lora_pilot
model: Qwen2.5-1.5B-Instruct
dataset: ToolACE baseline train 前 64 条
method: LoRA SFT pipeline validation
hyperparameter: rank=16, alpha=32, dropout=0.05, 5 steps, bf16
result: train_loss=0.3038, runtime=5.254s, adapter reload passed
analysis: 数据模板、LoRA 注入、反向传播和 adapter 保存链路均正常
next_step: 启动正式两轮训练
```

该实验仅验证训练链路，不纳入方法效果比较。

## 实验三：普通 LoRA SFT

```makefile
experiment_name: stage1_lora_baseline
model: Qwen2.5-1.5B-Instruct
dataset: ToolACE baseline_v1, train=1350, dev=150
method: supervised fine-tuning with LoRA
hyperparameter: rank=16, alpha=32, dropout=0.05, target=all, epoch=2, effective_batch=16, lr=1e-4, cosine, warmup=10, cutoff=2048, bf16
result: train_loss=0.3195, eval_loss=0.2662, Tool=93.88%, Arg=88.75%, Arg Exact=80.61%, controlled execution=79.17%
analysis: 格式、拒答和常见参数处理改善；未见 Schema 与多工具边界仍不稳定，且置信区间跨 0
next_step: 按错误类型构造困难样本，并进行 Schema 增强消融
```

训练信息：

- 训练步数：170
- 最佳检查点：`checkpoint-170`，按 `eval_loss` 选择
- 训练耗时：393.377 秒
- 训练吞吐：6.864 samples/s
- 可训练参数：18,464,768，占总参数 1.1820%
- 正式 adapter 大小：约 71 MiB
- adapter SHA-256：`dcf40498b1d5eb2459ae924438af362a1bd5b3e6851a9499698652ac40431b36`
- 适配器推理耗时：128.081 秒
- 推理峰值显存：3.067 GiB
- 预测 SHA-256：`15ab2dd9b22cb5b49fdae4e776e3b45f1390084c56ab272736bf8fedd5e5bb6c5`

## 核心对比

| Method | Model | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |
|---|---|---:|---:|---:|---:|
| Zero-shot | Qwen2.5-1.5B-Instruct | 91.84% | 86.90% | 76.53% | 79.17% |
| LoRA SFT | Qwen2.5-1.5B-Instruct | 93.88% | 88.75% | 80.61% | 79.17% |
| Proposed Method | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| GRPO（可选） | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

LoRA 相对 Zero-shot 的配对差值：

| 指标 | 差值 | 95% CI |
|---|---:|---:|
| Tool Accuracy | +2.04% | [-2.04%, +7.14%] |
| Arg Accuracy | +1.84% | [-3.06%, +6.92%] |
| Arg Exact | +4.08% | [-1.02%, +10.20%] |

## 分轨结果

| 轨道 | 方法 | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |
|---|---|---:|---:|---:|---:|
| controlled_v1 | Zero-shot | 96.43% | 90.18% | 78.57% | 79.17% |
| controlled_v1 | LoRA SFT | 96.43% | 90.18% | 82.14% | 79.17% |
| bfcl_lite_v1 | Zero-shot | 90.00% | 85.60% | 75.71% | N/A |
| bfcl_lite_v1 | LoRA SFT | 92.86% | 88.17% | 80.00% | N/A |

## 典型改善案例

1. 无关请求拒答：`bfcl_irrelevance_175` 和 `controlled_irrelevance_002` 从错误调用工具变为不调用。
2. 参数单位：`bfcl_simple_python_134` 将年收益率从 `7` 修正为 `0.07`。
3. 缺失参数：`bfcl_multiple_128` 补全 `dividends_paid`。
4. 时间格式：`controlled_argument_006` 去掉标准答案不接受的末尾 `Z`。
5. 并行格式：`bfcl_parallel_95` 从不可解析输出变为两个合法调用，但仍有参数错误。
6. 调用完整性：`bfcl_parallel_multiple_157` 补全第四个工具调用。

## 典型退化案例

1. 过调用：`bfcl_multiple_7` 原本正确调用一个种群评估工具，LoRA 额外调用了生态影响工具。
2. 未见 Schema 拒绝调用：`controlled_unseen_006` 原本能正确调用 `atlas_thread`，LoRA 后完全不调用。
3. 必填字段遗漏：`controlled_competition_002` 遗漏 `title`；`bfcl_parallel_multiple_93` 在两个调用中均遗漏 `calculate_years_ago`。
4. 多调用参数退化：`bfcl_parallel_multiple_15` 遗漏第二个餐厅搜索的 `rating`。

## 科研解释

### 为什么小模型仍会失败

小模型并非只缺少工具名称知识。错误同时来自意图边界、Schema 约束跟踪、并行规划和参数值规范化。尤其在多个相似工具竞争或同一工具多次调用时，模型需要同时保持调用集合与每个参数槽位，容易出现遗漏或额外调用。

### 数据质量如何影响泛化

普通 ToolACE 子集提升了常见调用模式和格式稳定性，但没有显式训练 Schema 重命名、必填字段扰动和困难负例，因此改善主要发生在同分布模式上。`controlled_unseen_006` 的退化说明，仅增加常规 SFT 数据可能强化熟悉模式，同时削弱对陌生命名的信心。

### SFT 与 RL 各自适合解决什么

SFT 适合建立输出格式、工具调用先验和常见参数映射。对于“调用集合是否完整”“必填字段是否全部满足”“执行结果是否成功”等可验证目标，小规模 RLVR/GRPO 更有潜力，但必须在 Stage 3 的 SFT 方法和奖励评测稳定后再尝试。

### 模型是在学习还是记忆

训练前已对冻结测试集执行工具名、完整 Schema、精确问题和字符 4-gram 高相似污染检查，未发现报告阈值内的重合；未见 Schema 轨道仍有 66.67% 参数完全匹配，说明模型具备一定的结构迁移能力。但当前样本量不足以排除更宽泛的模板记忆，Stage 3 必须增加 Schema 重命名消融与成对反事实测试。

## Stage 2/3 决策

后续方法暂定为“失败驱动的 Schema 鲁棒课程学习”，包含四类可独立消融的数据变换：

1. 工具名和参数名的一致重命名，测量未见 Schema 泛化。
2. 必填字段、参数顺序和描述表达扰动，强化 Schema 约束跟踪。
3. 相似工具困难负例与 no-tool 成对样本，控制过调用和拒答边界。
4. 并行调用拆分、合并和缺失调用反例，强化调用集合完整性。

在上述 SFT 方法完成消融前，不启动 GRPO，避免用奖励优化掩盖数据与评测问题。

## 产物索引

- 训练配置：`configs/stage1_lora_baseline.yaml`
- 训练日志：`logs/stage1_lora_baseline_console.log`
- 适配器：`checkpoints/stage1_lora_baseline/`
- 数据质量：`reports/stage1/data_quality_report.md`
- Zero-shot 明细：`reports/stage1/evaluation_v2/`
- LoRA 明细：`reports/stage1/lora_evaluation/`
- 配对统计：`reports/stage1/baseline_comparison.json` 与 `baseline_comparison.md`

所有失败样本均保留在逐样本结果中；未修改冻结测试集，也未依据结果选择性重跑。
