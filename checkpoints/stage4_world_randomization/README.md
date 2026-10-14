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
- name: stage4_world_randomization
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stage4_world_randomization

This model is a fine-tuned version of [/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct](https://huggingface.co//root/autodl-tmp/models/Qwen2.5-1.5B-Instruct) on the toolace_world_randomization_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2708

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
- seed: 20261017
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- training_steps: 177

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.1933        | 0.1412 | 25   | 0.3269          |
| 0.2663        | 0.2825 | 50   | 0.2983          |
| 0.2925        | 0.4237 | 75   | 0.2831          |
| 0.1966        | 0.5650 | 100  | 0.2757          |
| 0.2247        | 0.7062 | 125  | 0.2747          |
| 0.2687        | 0.8475 | 150  | 0.2713          |
| 0.3218        | 0.9887 | 175  | 0.2709          |
| 0.3218        | 1.0    | 177  | 0.2708          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2