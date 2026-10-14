---
library_name: peft
license: other
base_model: /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
tags:
- base_model:adapter:/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: stage3_schema_only
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stage3_schema_only

This model is a fine-tuned version of [/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct](https://huggingface.co//root/autodl-tmp/models/Qwen2.5-1.5B-Instruct) on the toolace_schema_only_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2698

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 20261014
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- training_steps: 170

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.3354        | 0.2283 | 25   | 0.3124          |
| 0.1557        | 0.4566 | 50   | 0.2886          |
| 0.2116        | 0.6849 | 75   | 0.2778          |
| 0.2281        | 0.9132 | 100  | 0.2748          |
| 0.1835        | 1.1370 | 125  | 0.2715          |
| 0.1804        | 1.3653 | 150  | 0.2698          |
| 0.2934        | 1.5479 | 170  | 0.2698          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2