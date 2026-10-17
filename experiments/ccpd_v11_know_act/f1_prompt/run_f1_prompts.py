#!/usr/bin/env python3
"""在冻结 F0 开发/测试集上运行四种最小原生工具调用提示。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


PROMPTS = {
    "A_native": (
        "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
        "否则不要调用工具。需要调用时只输出结构化工具调用。"
    ),
    "B_rejection_allowed": (
        "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
        "否则不要调用工具。需要调用时只输出结构化工具调用。"
        "如果当前提供的工具无法完整满足用户要求，拒绝执行是允许的；请明确拒绝，"
        "不要尝试生成违反当前参数约束的工具调用。"
    ),
    "C_fixed_rejection": (
        "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
        "否则不要调用工具。需要调用时只输出结构化工具调用。"
        "如果当前提供的工具无法完整满足用户要求，拒绝执行是允许的；不要生成不合法调用。"
        "拒绝时只输出：无法使用当前工具完成请求。"
    ),
    "D_contract_first": (
        "你是严谨的工具调用助手。回答前先在内部依据当前工具定义核对功能、必填参数以及"
        "enum、minimum、maximum、const 等参数约束。只有当前合同能够完整满足用户要求时才调用；"
        "否则明确拒绝且不要生成工具调用。需要调用时只输出结构化工具调用。不要输出判断过程。"
    ),
}


def parse_args() -> argparse.Namespace:
    """解析模型、数据、划分与输出参数。"""
    parser = argparse.ArgumentParser(description="运行 CCPD-v1.1 F1 最小提示控制")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="base")
    parser.add_argument(
        "--splits",
        default="dev,test_id,test_ood_family,test_ood_constraint",
    )
    parser.add_argument("--conditions", default=",".join(PROMPTS))
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    parser.add_argument("--resume-partial", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 出现空行")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_one(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    """执行一次确定性生成并返回响应、Token 数和延迟。"""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return (
        tokenizer.decode(new_tokens, skip_special_tokens=False).strip(),
        int(inputs["input_ids"].shape[1]),
        int(new_tokens.shape[0]),
        elapsed,
    )


def main() -> None:
    """加载模型并对固定条件、配对和两侧逐一生成。"""
    args = parse_args()
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = set(conditions) - set(PROMPTS)
    if unknown:
        raise ValueError(f"未知提示条件：{sorted(unknown)}")
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    pairs = [row for row in read_jsonl(args.pairs.resolve()) if row.get("split") in splits]
    if not pairs:
        raise ValueError("选定划分没有样本")
    expected_total = len(conditions) * len(pairs) * 2
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有 F1 预测：{args.output}")

    completed_keys: set[tuple[str, str, str]] = set()
    if partial.exists():
        if not args.resume_partial:
            raise FileExistsError(f"存在未完成预测；需显式传入 --resume-partial：{partial}")
        for row in read_jsonl(partial):
            key = (row["condition"], row["pair_id"], row["side"])
            if key in completed_keys:
                raise RuntimeError(f"Partial 中预测键重复：{key}")
            completed_keys.add(key)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path.resolve(), local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path.resolve(),
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter_path.resolve(), local_files_only=True)
        if args.adapter_path
        else base
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mode = "a" if partial.exists() else "x"
    with partial.open(mode, encoding="utf-8", newline="\n") as handle:
        for condition in conditions:
            for pair in pairs:
                for side_name in ("plus", "minus"):
                    key = (condition, pair["pair_id"], side_name)
                    if key in completed_keys:
                        continue
                    side = pair[side_name]
                    messages = [
                        {"role": "system", "content": PROMPTS[condition]},
                        *side["messages"],
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tools=side["tools"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    response, input_tokens, output_tokens, latency = generate_one(
                        model, tokenizer, prompt, args.max_new_tokens
                    )
                    row = {
                        "condition": condition,
                        "pair_id": pair["pair_id"],
                        "source_id": pair["source_id"],
                        "split": pair["split"],
                        "constraint_family": pair["constraint_family"],
                        "side": side_name,
                        "expected_decision": side["expected_decision"],
                        "response": response,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_seconds": round(latency, 6),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    completed_keys.add(key)
                    completed = len(completed_keys)
                    if completed == 1 or completed % 20 == 0 or completed == expected_total:
                        print(f"{args.model_name}: 已完成 {completed}/{expected_total}", flush=True)
    if len(completed_keys) != expected_total:
        raise RuntimeError(f"预测数量异常：{len(completed_keys)}/{expected_total}")
    partial.replace(args.output)
    metadata = {
        "model_name": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "pairs": len(pairs),
        "conditions": conditions,
        "splits": sorted(splits),
        "samples": expected_total,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt_texts": {name: PROMPTS[name] for name in conditions},
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0.0
        ),
        "pairs_sha256": sha256_file(args.pairs.resolve()),
        "predictions_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
