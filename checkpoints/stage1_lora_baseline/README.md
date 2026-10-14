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
- name: stage1_lora_baseline
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stage1_lora_baseline

This model is a fine-tuned version of [/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct](https://huggingface.co//root/autodl-tmp/models/Qwen2.5-1.5B-Instruct) on the toolace_baseline_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2662

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
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.3903        | 0.2959 | 25   | 0.3125          |
| 0.4782        | 0.5917 | 50   | 0.2867          |
| 0.3280        | 0.8876 | 75   | 0.2750          |
| 0.4348        | 1.1775 | 100  | 0.2699          |
| 0.3876        | 1.4734 | 125  | 0.2675          |
| 0.3385        | 1.7692 | 150  | 0.2665          |
| 0.1926        | 2.0    | 170  | 0.2662          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2