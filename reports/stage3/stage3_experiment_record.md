# Stage 3 方法与消融实验记录

## 方法名称

**失败驱动的 Schema 鲁棒课程学习**

英文：**Failure-Driven Schema-Robust Curriculum（FDSRC）**

该方法只使用已清洗训练集进行确定性增强，不调用外部大模型生成标签，也不读取测试答案。核心增强包括：

1. 对工具名和顶层参数名做一致、无语义别名化，迫使模型依赖描述和 Schema 结构。
2. 对工具、参数属性和必填字段顺序做语义不变的排列扰动。
3. 按用户请求与候选工具描述的词元重合度选择困难 no-tool 样本，强化拒答边界。
4. 对多调用样本做独立别名化，强化调用集合完整性。

## 实验纪律

- Baseline、Schema-only 和完整方法均固定 170 个优化步、有效 batch 16、相同学习率与 LoRA 配置。
- `frozen_v1` 用于错误分析和方法开发，后续明确视为开发诊断集。
- 方法定型后，从 BFCL 未使用样本中冻结 70 条独立 `final_holdout_v1`，与开发集 source ID 完全互斥。
- 四种方法在最终 holdout 上只运行一次，不依据结果修改方法。
- 完整保留训练日志、检查点、预测、逐样本结果和失败案例。

## 数据构建

| 数据集 | 原始样本 | Schema 别名 | 困难 no-tool | 多调用增强 | 最终样本 |
|---|---:|---:|---:|---:|---:|
| Schema-only | 1,350 | 400 | 0 | 0 | 1,750 |
| FDSRC | 1,350 | 400 | 180 | 546 | 2,429 |

因完整内容去重，FDSRC 最终数量小于各来源简单相加。

- Schema-only SHA-256：`720736bbae0b4aed5de88d4244b14746de49d4041c21863bdcaa382c15192c72`
- FDSRC SHA-256：`73c7ed0d30dc249bf279f0e4c6aa2abcf47bd0db6221e19970c4a47c25281afd`
- 最终 holdout 清单 SHA-256：`ea632c0ab8f1a126b041b4a9a2adbfaeeb2b12ccd4a244b63f68b99bb15be044`

## 实验一：Schema-only 消融

```makefile
experiment_name: stage3_schema_only
model: Qwen2.5-1.5B-Instruct
dataset: baseline train 1350 + schema alias 400; shared dev 150
method: LoRA SFT with consistent schema aliasing
hyperparameter: rank=16, alpha=32, dropout=0.05, all-linear, max_steps=170, effective_batch=16, lr=1e-4, bf16
result: train_loss=0.2650, eval_loss=0.2698; final holdout Tool=98.57%, Arg=91.56%, Arg Exact=77.14%
analysis: 工具选择和参数字段级准确率提高，但参数完全匹配下降，说明仅做别名化不足以保持全部参数槽位
next_step: 加入困难 no-tool 与多调用完整性增强
```

- 训练耗时：461.722 秒
- 最佳检查点：`checkpoint-170`
- adapter SHA-256：`fb818ba63d221c5d78f4f8db135d70082a1a645245de7dd4b5794328e31b4264d`

## 实验二：完整 FDSRC

```makefile
experiment_name: stage3_failure_driven
model: Qwen2.5-1.5B-Instruct
dataset: baseline train 1350 + schema alias + hard no-tool + multi-call augmentation; shared dev 150
method: Failure-Driven Schema-Robust Curriculum with LoRA SFT
hyperparameter: rank=16, alpha=32, dropout=0.05, all-linear, max_steps=170, effective_batch=16, lr=1e-4, bf16
result: train_loss=0.2601, eval_loss=0.2674; final holdout Tool=100.00%, Arg=93.94%, Arg Exact=81.43%
analysis: 独立 holdout 三项总体指标均为最高；消除了工具选择错误，但隐含参数和复杂参数值仍会出错
next_step: 扩大独立测试集，补充默认参数与数值尺度增强，再决定是否进行小规模 RLVR
```

- 训练耗时：490.884 秒
- 最佳检查点：`checkpoint-170`
- adapter SHA-256：`e2ee428936015e57d2eb9919c76266dd80fee68a5fa9f88d0f81be5c67f6ffda4`

## 开发诊断集结果

| Method | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |
|---|---:|---:|---:|---:|
| Zero-shot | 91.84% | 86.90% | 76.53% | 79.17% |
| LoRA SFT | 93.88% | 88.75% | 80.61% | 79.17% |
| Schema-only | 93.88% | 88.83% | 79.59% | 83.33% |
| FDSRC | **94.90%** | **89.77%** | 80.61% | **83.33%** |

开发集被用于方法设计，只能用于诊断与消融，不作为最终泛化证据。

## 独立最终 Holdout

| Method | Model | Tool Accuracy | Arg Accuracy | Arg Exact |
|---|---|---:|---:|---:|
| Zero-shot | Qwen2.5-1.5B-Instruct | 95.71% | 90.25% | 80.00% |
| LoRA SFT | Qwen2.5-1.5B-Instruct | 97.14% | 91.14% | 78.57% |
| Schema-only | Qwen2.5-1.5B-Instruct | 98.57% | 91.56% | 77.14% |
| **FDSRC** | Qwen2.5-1.5B-Instruct | **100.00%** | **93.94%** | **81.43%** |
| GRPO（可选） | 未执行 | N/A | N/A | N/A |

FDSRC 相对普通 LoRA 的配对差值：

| 指标 | 差值 | 95% 配对 bootstrap CI |
|---|---:|---:|
| Tool Accuracy | +2.86% | [0.00%, +7.14%] |
| Arg Accuracy | +2.80% | [-0.71%, +7.14%] |
| Arg Exact | +2.86% | [-2.86%, +8.57%] |

FDSRC 相对 Zero-shot 的配对差值：Tool `+4.29%`，Arg `+3.69%`，Arg Exact `+1.43%`。由于置信区间触及或跨过 0，本项目报告一致领先趋势，不宣称统计显著。

## 分类别观察

- Irrelevance：FDSRC 为 100%，Zero-shot 为 70%，说明困难 no-tool 数据显著改善拒答边界。
- Multiple：FDSRC Arg Exact 为 90%，四种方法中最高。
- Parallel Multiple：FDSRC Tool Accuracy 为 100%、Arg Exact 为 60%，优于普通 LoRA 的 80% 和 50%。
- Parallel：所有 SFT 方法 Arg Exact 均为 60%，低于 Zero-shot 的 80%，说明当前增强没有解决复杂并行参数值问题。
- Simple Python：FDSRC Arg Exact 为 85%，仍低于 Zero-shot 的 95%，常见函数上的参数默认值可能被 SFT 分布扰动。

## 剩余失败类型

完整方法在 holdout 上不再出现工具选择错误，但仍有 13 条参数错误：

1. 隐含默认参数遗漏，例如 `detail=true`、`live_conversion=true`、输出单位和热量上限。
2. 数值尺度混淆，例如利率 `4` 与 `0.04`、人口数量少一个数量级。
3. 复杂结构错误，例如卡方表应为一维数组却生成二维数组。
4. 指代与状态跟踪错误，例如连续行程把 `office`、`home` 当作字面地址。
5. 多调用中的可选字段不一致，例如多个相似调用只漏掉其中一个 `temp_unit`。

## 是否启动 GRPO

本轮不启动 GRPO。原因不是训练链路受限，而是当前主要剩余问题集中于参数语义、默认值和结构表示；在 70 条 holdout 下，SFT 改进尚未达到统计显著。此时加入 RL 会增加归因难度。下一轮应先扩充不参与调参的评测集，并为参数级、Schema 级与执行级奖励分别做单元测试，再进行 50 至 100 步小规模 GRPO。

## 产物

- 数据构建：`scripts/build_robust_sft.py`
- 独立 holdout 构建：`scripts/build_final_holdout.py`
- 配置：`configs/stage3_schema_only.yaml`、`configs/stage3_failure_driven.yaml`
- 数据：`data/sft/schema_only_v1/`、`data/sft/failure_driven_v1/`
- 模型：`checkpoints/stage3_schema_only/`、`checkpoints/stage3_failure_driven/`
- 日志：`logs/stage3_*.log`、`logs/final_*.log`
- 开发集结果：`reports/stage3/`
- 独立结果：`reports/final_holdout/`
