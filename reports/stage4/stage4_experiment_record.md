# Stage 4 动态工具环境泛化实验记录

## 研究决策

已有 FDSRC 在固定 70 条 holdout 上达到 100% Tool Accuracy，但在 750 条工具世界诊断集的 `capability_substitution` 上仅有 0.67% Decision Exact。因此选择 **Environment-Conditioned Tool Learning**，没有转向 GRPO 或堆叠复杂模块。

提出的方法为：**Failure-Weighted Counterfactual Tool-World SFT（FWCT-SFT）**。

核心假设：标准 SFT 不会因同一意图的样本成对出现而自动学习跨世界约束；需要依据诊断失败率提高关键反事实世界的监督密度。

## 数据与污染控制

- 方法选择集：150 个 BFCL 基础任务，5 个世界，共 750 条，种子 `20261016`。
- Stage 4 训练：排除旧开发集、旧最终集和方法选择集的 290 个 source ID。
- 独立最终集：60 个新基础任务，5 个世界，共 300 条，种子 `20261019`。
- 最终构建时重建并排除全部 Stage 4 BFCL 训练来源；总计排除 756 个 source ID。
- 最终数据 SHA-256：`a40b6d464c74b39a9c7fccdcd8a68fe02599aa1840896f1d1cf9cf3f2793c4e9`。
- 因严格去重后 `multiple` 类仅余 2 条，最终类别分布固定为 `28/2/15/15`，未放宽污染规则。

## 统一训练配置

```makefile
model: Qwen2.5-1.5B-Instruct
framework: LLaMA-Factory 0.9.5.dev0
method: LoRA SFT
lora: rank=16, alpha=32, dropout=0.05, target=all-linear
precision: BF16
effective_batch: 16
learning_rate: 1e-4
max_steps: 177
cutoff_len: 2048
base_dataset: FDSRC 2429 samples
added_samples: 400
total_train_samples: 2829
dev_samples: 150
```

## 实验一：World Randomization

```makefile
experiment_name: stage4_world_randomization
model: Qwen2.5-1.5B-Instruct
dataset: FDSRC 2429 + 400 unpaired tool-world samples
method: 独立意图的普通工具环境随机增强
hyperparameter: 统一 Stage 4 配置，seed=20261017
result: train_loss=0.2421, eval_loss=0.2708, diagnostic Decision Exact=60.00%, final=55.33%
analysis: 明显改善接口与干扰世界，但能力替换仍只有 diagnostic 3.33%、final 3.33%
next_step: 与同意图反事实配对进行等预算对照
```

- 新增独立意图：400。
- 训练耗时：489.741 秒。
- 训练数据 SHA-256：`ff5b55892c92c375db69c02a11a50a369477f1415bdc2e9a3bd801b2316e51f6`。

## 实验二：Counterfactual Paired

```makefile
experiment_name: stage4_counterfactual_paired
model: Qwen2.5-1.5B-Instruct
dataset: FDSRC 2429 + 200 original/transformed pairs
method: 同一意图的原始世界与一个反事实世界共同监督
hyperparameter: 统一 Stage 4 配置，seed=20261017
result: train_loss=0.2443, eval_loss=0.2720, diagnostic Decision Exact=58.67%, final=54.00%
analysis: 相对随机增强 diagnostic -1.33 pp、final -1.33 pp；标准 SFT 未利用配对关系，意图覆盖减半可能带来代价
next_step: 保留负结果，验证失败类型监督密度而非配对本身
```

- 新增独立意图：200。
- 训练耗时：494.779 秒。
- 训练数据 SHA-256：`98e670ccca903212c144063c79840c300dc15c2c88f338302b49df6fedf9b9f6`。

## 实验三：FWCT-SFT

```makefile
experiment_name: stage4_failure_weighted
model: Qwen2.5-1.5B-Instruct
dataset: FDSRC 2429 + original 150 + substitution 150 + removal 50 + interface 25 + distractor 25
method: Failure-Weighted Counterfactual Tool-World SFT
hyperparameter: 统一 Stage 4 配置，seed=20261018
result: train_loss=0.2439, eval_loss=0.2729, diagnostic Decision Exact=64.13%, final=62.33%
analysis: 能力替换 diagnostic 47.33%、final 56.67%；总体显著提升，但干扰世界出现拒绝偏置
next_step: 研究保持等预算的多目标采样，恢复 distractor 而不损失 substitution
```

- 新增独立意图：250。
- 训练耗时：498.251 秒。
- 训练数据 SHA-256：`e76038016cf0ef23256b930b91451a7b9c599b664fa4829eeb5c9a4c0069aa72`。
- 最佳验证损失来自 `checkpoint-175`，最终 adapter 已加载该最佳权重。

## 方法选择集结果

| 方法 | Overall | Original | Interface | Removal | Substitution | Distractor | All-worlds |
|---|---:|---:|---:|---:|---:|---:|---:|
| FDSRC | 54.67% | 74.67% | 61.33% | 74.67% | 0.67% | 62.00% | 0.67% |
| World Randomization | 60.00% | 76.00% | 70.67% | 76.67% | 3.33% | 73.33% | 2.67% |
| Counterfactual Paired | 58.67% | 76.00% | 67.33% | 75.33% | 0.67% | 74.00% | 0.67% |
| FWCT-SFT | 64.13% | 72.67% | 64.67% | 77.33% | 47.33% | 58.67% | 14.00% |

FWCT-SFT 相对 FDSRC 总体 `+9.47 pp`，95% CI `[+6.40,+12.53]`。相对 World Randomization 总体 `+4.13 pp`，CI `[+1.20,+7.07]`；Substitution `+44.00 pp`，Distractor `-14.67 pp`。

## 独立最终结果

| 方法 | Tool Acc. | Arg Acc. | Arg Exact | Decision Exact | Substitution | Distractor |
|---|---:|---:|---:|---:|---:|---:|
| FDSRC | 70.33% | 63.63% | 52.67% | 52.67% | 0.00% | 60.00% |
| World Randomization | 73.00% | 66.90% | 55.33% | 55.33% | 3.33% | 65.00% |
| Counterfactual Paired | 72.33% | 65.91% | 54.00% | 54.00% | 1.67% | 66.67% |
| FWCT-SFT | 78.67% | 73.08% | 62.67% | 62.33% | 56.67% | 53.33% |

FWCT-SFT 相对 FDSRC 总体 `+9.67 pp`，95% CI `[+4.67,+14.67]`；相对 World Randomization `+7.00 pp`，CI `[+2.33,+11.33]`。Paired 相对 Randomization 为 `-1.33 pp`，CI `[-3.67,+1.00]`。

## 中断与恢复记录

最终四路推理第一次运行因 SSH 会话中断，保留 `.partial`，没有删除失败产物。推理器随后增加显式 `--resume-partial`：

| 方法 | 已校验并复用 | 本次补齐 | 本次进程耗时 |
|---|---:|---:|---:|
| FDSRC | 216 | 84 | 186.382 秒 |
| World Randomization | 191 | 109 | 209.791 秒 |
| Counterfactual Paired | 208 | 92 | 186.987 秒 |
| FWCT-SFT | 271 | 29 | 54.723 秒 |

恢复模式拒绝重复 ID 和评测集外 ID，并在预测元数据中声明耗时只覆盖恢复进程，不估算中断前耗时。

## 产物索引

- 工具世界构造：`scripts/build_tool_world_eval.py`
- Stage 4 数据：`scripts/build_counterfactual_sft.py`、`scripts/build_failure_weighted_sft.py`
- 独立最终集：`scripts/build_tool_world_final.py`
- 严格指标：`evaluation/tool_world_metrics.py`
- 推理恢复：`scripts/run_adapter_eval.py`
- 比较统计：`scripts/compare_tool_world_methods.py`
- 方法选择比较：`reports/tool_world/comparison/diagnostic_comparison.md`
- 独立最终比较：`reports/tool_world_final/comparison/final_comparison.md`
- 完整研究报告：`reports/final_report.md`
