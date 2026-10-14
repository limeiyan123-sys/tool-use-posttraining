# 面向工具调用泛化的小参数语言模型后训练优化框架

英文名称：**Post-training Optimization Framework for Generalizable Tool-Using Small Language Models**

本项目研究有限算力下，`Qwen2.5-1.5B-Instruct` 能否根据当前工具 Schema 与实际能力变化进行条件化决策。项目不是只跑一次开源训练，而是完整执行了“诊断、假设、等预算对照、独立验证”的研究闭环。

## 核心结论

固定 BFCL 接口上的高准确率不能证明动态工具环境泛化。FDSRC 在旧 70 条 holdout 上达到 100% Tool Accuracy，但在 750 条 Counterfactual Tool-World 诊断集的能力替换世界中，Decision Exact 仅为 0.67%。

本项目据此提出 **Failure-Weighted Counterfactual Tool-World SFT（FWCT-SFT）**：不增加训练样本数和优化步数，而是按真实诊断失败率提高能力替换反事实监督密度。方法冻结后，在 300 条独立工具世界测试集上得到：

| 方法 | Overall Decision Exact | Original | Interface | Removal | Substitution | Distractor |
|---|---:|---:|---:|---:|---:|---:|
| FDSRC | 52.67% | 68.33% | 55.00% | 80.00% | 0.00% | 60.00% |
| World Randomization | 55.33% | 71.67% | 56.67% | 80.00% | 3.33% | 65.00% |
| Counterfactual Paired | 54.00% | 68.33% | 53.33% | 80.00% | 1.67% | 66.67% |
| **FWCT-SFT** | **62.33%** | 70.00% | 51.67% | 80.00% | **56.67%** | 53.33% |

FWCT-SFT 相对 FDSRC 的总体提升为 `+9.67 pp`，95% 成对 Bootstrap 区间为 `[+4.67,+14.67] pp`；相对 World Randomization 为 `+7.00 pp`，区间为 `[+2.33,+11.33] pp`。收益几乎全部来自能力替换，代价是干扰世界相对随机增强下降 `11.67 pp`。因此，本项目将它表述为针对能力约束服从的有效专门化，而不是全面占优的方法。

## 研究设计

### 工具世界

每个基础请求确定性生成五个成组环境：

1. `original`：原始环境。
2. `ephemeral_interface`：工具名和参数名匿名化、顺序扰动，但语义不变。
3. `capability_removal`：移除关键工具，同时保留仍可独立完成的调用。
4. `capability_substitution`：用参数约束不兼容的近似工具替换关键工具，并重新计算 Gold。
5. `distractor`：加入描述相似但能力不适用的工具。

评测器严格解析模型输出，不修复非法 JSON，不猜测缺失字段。除 Tool Accuracy 和 Arg Exact 外，还计算 Interface Equivariance、Counterfactual Pair Accuracy 与五世界全对准确率。

### 等预算训练

三种 Stage 4 方法均使用 2,429 条 FDSRC 基础数据加 400 条工具世界样本，总计 2,829 条；LoRA 配置、有效 batch、学习率和 177 个优化步完全一致。

- `World Randomization`：400 个互异意图，原始与变换环境不配对。
- `Counterfactual Paired`：200 个意图，每个意图同时提供原始与一个变换环境。
- `FWCT-SFT`：250 个意图，将 150 条能力替换样本与其原始环境配对，并保留少量其他世界监督。

标准配对 SFT 未优于随机增强，说明“同意图成对出现”本身不会自动产生跨世界约束学习；失败加权的关键样本密度才是本轮有效因素。

## 目录

```text
tool-use-posttraining/
├── configs/                  # 训练与评测配置
├── data/
│   ├── evaluation/           # 冻结诊断集、独立最终集及清单
│   └── sft/                  # 数据清单；训练正文不公开提交
├── evaluation/               # 严格解析、Schema 校验、工具世界指标
├── reports/                  # 实验账本、汇总、逐样本结果与报告
├── scripts/                  # 数据构建、推理、评测、统计比较
├── tests/                    # 23 项单元测试
├── checkpoints/README.md     # 权重获取与重训说明
├── environment_report.md
└── README.md
```

## 关键复现命令

```bash
source /root/autodl-tmp/envs/tooluse/bin/activate

# 构建 750 条方法选择诊断集
python scripts/build_tool_world_eval.py \
  --project-root /root/autodl-tmp/tool-use-posttraining \
  --output-dir data/evaluation/tool_world_v1

# 构建三组等预算 Stage 4 数据
python scripts/build_counterfactual_sft.py --project-root .
python scripts/build_failure_weighted_sft.py --project-root .

# 训练提出的方法
llamafactory-cli train configs/stage4_failure_weighted.yaml

# 推理与严格评测
python scripts/run_adapter_eval.py --config configs/tool_world_failure_weighted.yaml
python scripts/evaluate_tool_world_predictions.py \
  --project-root . \
  --frozen-dir data/evaluation/tool_world_v1 \
  --predictions reports/tool_world/failure_weighted_predictions.jsonl \
  --output-dir reports/tool_world/failure_weighted_evaluation \
  --run-name failure_weighted

# 方法冻结后构建独立最终集
python scripts/build_tool_world_final.py \
  --project-root . \
  --output-dir data/evaluation/tool_world_final_v1

python -m unittest discover -s tests -v
```

完整研究叙事见 `reports/final_report.md`，Stage 4 实验账本见 `reports/stage4/stage4_experiment_record.md`，机器可读的成对统计见 `reports/tool_world_final/comparison/final_comparison.json`。

## 发布边界

GitHub 版本不包含 LoRA 权重、第三方仓库快照、完整终端日志和 SFT 正文。上游训练数据中存在形似访问令牌的示例字符串，因此只公开数据构建代码、清单与 SHA-256。服务器保留全部 Checkpoint、训练数据和原始日志。

BFCL 子集结果不是官方排行榜成绩；公开基准也可能出现在基础模型预训练数据中。最终集只有 60 个基础任务，且严格去重后 `multiple` 类仅剩 2 条，结论仍需在更大规模和真实执行环境中复核。
