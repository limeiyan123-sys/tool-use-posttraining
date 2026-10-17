#!/usr/bin/env python3
"""在 E2 合同配对 Pilot 上运行确定性原生工具调用。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析模型、Adapter 与输出参数。"""
    parser = argparse.ArgumentParser(description="运行 CCPD E2 原生工具调用")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.16)
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


def direct_messages(side: dict[str, Any]) -> list[dict[str, str]]:
    """构造与历史 Stage 6 一致的原生决策消息。"""
    system = {
        "role": "system",
        "content": (
            "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
            "否则不要调用工具。需要调用时只输出结构化工具调用。"
        ),
    }
    return [system, *side["messages"]]


def generate_one(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    """执行一次贪心生成并返回原始响应。"""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def main() -> None:
    """加载一个模型并依次运行配对两侧的原生生成。"""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有预测：{args.output}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.cuda.reset_peak_memory_stats()
    pairs = read_jsonl(args.pairs.resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path.resolve(), local_files_only=True, trust_remote_code=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path.resolve(),
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = (
        PeftModel.from_pretrained(base_model, args.adapter_path.resolve(), local_files_only=True)
        if args.adapter_path
        else base_model
    )
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    started = time.perf_counter()
    completed = 0
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            for side_name in ("plus", "minus"):
                side = pair[side_name]
                prompt = tokenizer.apply_chat_template(
                    direct_messages(side), tools=side["tools"], tokenize=False, add_generation_prompt=True
                )
                response = generate_one(model, tokenizer, prompt, args.max_new_tokens)
                row = {
                    "pair_id": pair["pair_id"],
                    "side": side_name,
                    "expected_decision": side["expected_decision"],
                    "response": response,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                completed += 1
                if completed == 1 or completed % 20 == 0 or completed == len(pairs) * 2:
                    print(f"{args.model_name}: 已完成 {completed}/{len(pairs) * 2}", flush=True)
    partial.replace(args.output)
    metadata = {
        "model_name": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "pairs": len(pairs),
        "samples": len(pairs) * 2,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
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
