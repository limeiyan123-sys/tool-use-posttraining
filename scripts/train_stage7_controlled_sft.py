#!/usr/bin/env python3
"""使用 Transformers 与 PEFT 执行 Stage 7 最小受控 LoRA SFT。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = (
    "你是严谨的工具调用助手。必须依据本轮提供的工具说明选择能够完整满足请求的工具；"
    "不要依赖工具过去的功能。需要调用时只输出结构化工具调用。"
)


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="Stage 7 最小受控 LoRA SFT")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.20)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tools(raw_tools: str) -> list[dict[str, Any]]:
    """将 LLaMA-Factory 工具字段转换为 OpenAI 工具包装格式。"""
    functions = json.loads(raw_tools)
    if not isinstance(functions, list) or not functions:
        raise ValueError("tools 必须是非空函数列表")
    return [{"type": "function", "function": function} for function in functions]


def encode_record(
    record: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """编码单条样本，只监督 assistant 工具调用响应。"""
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError("每条受控 SFT 样本必须只有 human 与 function_call 两轮")
    human, function_call = conversations
    if human.get("from") != "human" or function_call.get("from") != "function_call":
        raise ValueError("受控 SFT 轮次类型不符合预期")
    tools = normalize_tools(record["tools"])
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": human["value"]},
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    call = json.loads(function_call["value"])
    if set(call) != {"name", "arguments"}:
        raise ValueError("function_call 必须且只能包含 name 与 arguments")
    response = (
        "<tool_call>\n"
        + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
        + tokenizer.eos_token
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    if len(input_ids) > max_length:
        raise ValueError(f"样本长度 {len(input_ids)} 超过 max_length={max_length}")
    if not response_ids or all(label == -100 for label in labels):
        raise RuntimeError("样本没有可监督 Token")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class ToolSftDataset(Dataset):
    """保存已经严格编码的工具调用监督样本。"""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        """预编码全部样本，以便训练前暴露格式错误。"""
        self.examples = [encode_record(row, tokenizer, max_length) for row in rows]

    def __len__(self) -> int:
        """返回样本数。"""
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """返回单条编码样本。"""
        return self.examples[index]


@dataclass
class ToolSftCollator:
    """动态填充输入，并使用 -100 填充监督标签。"""

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """将不同长度样本整理为批次张量。"""
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def dataset_audit(dataset: ToolSftDataset) -> dict[str, Any]:
    """汇总训练前可审计的长度与监督 Token 信息。"""
    lengths = [len(example["input_ids"]) for example in dataset.examples]
    supervised = [sum(label != -100 for label in example["labels"]) for example in dataset.examples]
    return {
        "samples": len(dataset),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "min_supervised_tokens": min(supervised),
        "max_supervised_tokens": max(supervised),
    }


def main() -> None:
    """加载数据与基座，执行一次受控 LoRA SFT。"""
    args = parse_args()
    model_path = args.model_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("受控 LoRA 训练需要可用 CUDA GPU")
    if not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("cuda-memory-fraction 必须位于 (0, 1] 区间")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    print(
        f"[INFO] GPU={torch.cuda.get_device_name(0)}，"
        f"显存使用上限={args.cuda_memory_fraction:.0%}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ToolSftDataset(read_json(dataset_dir / "train.json"), tokenizer, args.max_length)
    eval_dataset = ToolSftDataset(read_json(dataset_dir / "dev.json"), tokenizer, args.max_length)
    audit = {"train": dataset_audit(train_dataset), "dev": dataset_audit(eval_dataset)}
    print(json.dumps({"dataset_audit": audit}, ensure_ascii=False, indent=2))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=6,
        save_strategy="steps",
        save_steps=6,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ToolSftCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "dataset_audit": audit,
        "train_metrics": train_result.metrics,
        "eval_metrics": trainer.evaluate(),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target": "all-linear"},
    }
    (output_dir / "stage7_training_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
