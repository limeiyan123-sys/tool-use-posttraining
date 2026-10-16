# Stage 8 实验记录

experiment_name: stage8_learning_bottleneck

model: Qwen2.5-1.5B-Instruct 的 FDSRC、FWCT、FCTW 历史检查点

dataset: tool_world_final_v1，60 个基础任务、每个任务 5 个工具世界，共 300 条

method: 复用冻结逐样本预测，执行 Schema 过滤前后多标签错误归因与 Original/Ephemeral 配对诊断

hyperparameter: 无训练超参数；配对精确二项检验使用双侧检验；不合并三个模型作为独立样本

result: Schema 过滤后 Decision Exact 为 67.33%～68.00%；参数值绑定与必要条件遗漏占剩余错误的 54.17%～64.95%。等价接口改名后，三模型准确率下降 13.33～18.33 个百分点。

analysis: 主要残留是形式合法但语义错误的参数绑定。13/300 个 Gold 因可选受限参数及默认行为不明确被标为中等可信度，未用于高置信度机制结论。

next_step: 不直接训练新方法。先在新的冻结小样本上比较顺序控制、显式标识映射与 PA-Tool 类 Schema 对齐；若成熟基线已消除差距，则结束该 Schema 子方向。

cost: 未新增模型推理、未使用 GPU、未训练新模型，计算成本仅为 CPU 离线统计。
