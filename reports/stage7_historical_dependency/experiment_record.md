# Stage 7 实验记录

```text
experiment_name: stage7_historical_tool_dependency
model: Qwen2.5-1.5B-Instruct
dataset: historical_dependency_v1（24 个配对任务，48 个环境）
method: Base / 历史 LoRA / FDSRC / Original-only 受控 LoRA SFT
hyperparameter: LoRA r=8, alpha=16, dropout=0.05, lr=1e-4, epoch=4, seed=20261016
result: 四个模型 Original=100%，Updated=100%，Pair Success=100%，Stale Choice=0/24
analysis: 当前清晰工具契约覆盖了历史映射，H1 未获支持；Stage 6 失败更可能来自细粒度 Schema 语义绑定与决策整合。
next_step: 转向 Requirement-to-Schema Binding 的独立冻结评测，不启动 7B 或额外训练。
```

## 运行资源

- 服务器：Ubuntu 22.04，共享 2 × RTX 4090 D 48GB。
- 基础、普通 LoRA、FDSRC 推理：CPU，避免影响共享 GPU。
- 受控训练与最终推理：GPU 0，显存硬限制 20%，训练 43.65 秒，推理峰值 2.968 GiB。
- GPU 1 未使用。
- 新增付费资源：0。

## 环境限制

- 服务器已有 LLaMA-Factory 命令入口不可用，因此使用同一环境中的 Transformers 4.57.1、PEFT 0.17.1 与 PyTorch 2.10.0 完成最小训练。
- 历史 adapters 由更高版本 PEFT 保存，加载时对新增且与核心 LoRA 矩阵无关的配置字段产生 warning；未使用静默 fallback。
- 未运行顺序交换控制与 7B，因为所有模型均无错误，停止条件 C 已满足。
