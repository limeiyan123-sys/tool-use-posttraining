---
library_name: peft
license: other
base_model: /home/limeiyan/models/Qwen2.5-1.5B-Instruct
tags:
- base_model:adapter:/home/limeiyan/models/Qwen2.5-1.5B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: ccpd_v11_f4_balanced_sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ccpd_v11_f4_balanced_sft

This model is a fine-tuned version of [/home/limeiyan/models/Qwen2.5-1.5B-Instruct](https://huggingface.co//home/limeiyan/models/Qwen2.5-1.5B-Instruct) on the f4_balanced_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0264

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
- train_batch_size: 1
- eval_batch_size: 1
- seed: 20261017
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 5
- num_epochs: 4.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0334        | 0.5   | 20   | 0.0727          |
| 0.0475        | 1.0   | 40   | 0.0579          |
| 0.037         | 1.5   | 60   | 0.0434          |
| 0.0315        | 2.0   | 80   | 0.0363          |
| 0.0172        | 2.5   | 100  | 0.0328          |
| 0.0349        | 3.0   | 120  | 0.0264          |
| 0.0261        | 3.5   | 140  | 0.0268          |
| 0.0146        | 4.0   | 160  | 0.0271          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.57.1
- Pytorch 2.8.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2