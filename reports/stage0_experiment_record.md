# Stage 0 实验记录

```yaml
experiment_name: stage0_qlora_smoke
model: Qwen2.5-1.5B-Instruct
dataset: 4 条合成工具调用链路测试数据
method: bitsandbytes 4-bit QLoRA smoke test
hyperparameter:
  max_steps: 2
  batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 0.0001
  lora_rank: 8
  lora_alpha: 16
  compute_dtype: bfloat16
result:
  status: success
  step_1_loss: 0.1101201549
  step_2_loss: 0.1050859988
  train_loss: 0.1076030768
  train_runtime_seconds: 1.233
analysis: 训练、保存和 NF4 adapter 重载链路均通过；样本规模过小，不允许解释模型效果。
next_step: 建立冻结的 Zero-shot 工具调用评测协议与测试集。
```

## Warning

- 量化训练建议开启 `upcast_layernorm`。
- 正式配置需要替换已弃用的 `warmup_ratio` 与 `logging_dir` 参数。
- 本次没有验证集，因此不存在验证准确率或 eval loss。
- 本记录不属于正式实验对比表。
