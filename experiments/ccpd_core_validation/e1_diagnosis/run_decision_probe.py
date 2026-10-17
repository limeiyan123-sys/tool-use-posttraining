#!/usr/bin/env python3
"""运行双标签顺序的单 Token CALL/DECLINE 受控决策探针。"""

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
    parser = argparse.ArgumentParser(description="运行 CCPD E1 决策分数探针")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.16)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL 文件。"""
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


def messages_for(sample: dict[str, Any], reverse: bool) -> list[dict[str, str]]:
    """构造冻结的双顺序二选一诊断消息。"""
    mapping = "A = DECLINE，B = CALL" if reverse else "A = CALL，B = DECLINE"
    tool_json = json.dumps(sample["tools"][0]["function"], ensure_ascii=False, sort_keys=True)
    system = (
        "你正在完成一个受控工具决策测试。请依据候选工具的实际功能和参数约束，判断它能否完整满足用户请求。"
        "如果能完整满足则选择 CALL，否则选择 DECLINE。不要调用工具，不要解释。"
        f"本题标签映射为：{mapping}。最终只回答一个大写字母 A 或 B。"
    )
    user = f"用户请求：\n{sample['messages'][0]['content']}\n\n唯一候选工具：\n{tool_json}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def contextual_label_id(tokenizer: Any, prompt: str, label: str) -> int:
    """确认标签在当前生成上下文中恰好追加一个 Token。"""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(prompt + label, add_special_tokens=False)["input_ids"]
    if continuation_ids[: len(prompt_ids)] != prompt_ids or len(continuation_ids) != len(prompt_ids) + 1:
        raise RuntimeError(f"标签 {label!r} 在当前上下文中不是恰好一个追加 Token")
    return int(continuation_ids[-1])


def score_prompt(model: Any, tokenizer: Any, prompt: str) -> tuple[float, float, int, int]:
    """返回下一 Token 的 A/B Logit 及其 Token ID。"""
    import torch

    label_a = contextual_label_id(tokenizer, prompt, "A")
    label_b = contextual_label_id(tokenizer, prompt, "B")
    if label_a == label_b:
        raise RuntimeError("A/B 标签被编码为同一 Token")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    return float(logits[label_a]), float(logits[label_b]), label_a, label_b


def iter_samples(pairs: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    """按稳定顺序展开配对的有效侧与无效侧。"""
    samples: list[tuple[str, str, dict[str, Any]]] = []
    for pair in pairs:
        samples.append((pair["base_id"], "valid", pair["valid_sample"]))
        samples.append((pair["base_id"], "invalid", pair["invalid_sample"]))
    return samples


def main() -> None:
    """加载一个模型并完成冻结决策探针。"""
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
    rows: list[dict[str, Any]] = []
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        for index, (base_id, side, sample) in enumerate(iter_samples(pairs), 1):
            forward_prompt = tokenizer.apply_chat_template(
                messages_for(sample, reverse=False), tokenize=False, add_generation_prompt=True
            )
            reverse_prompt = tokenizer.apply_chat_template(
                messages_for(sample, reverse=True), tokenize=False, add_generation_prompt=True
            )
            forward = score_prompt(model, tokenizer, forward_prompt)
            reverse = score_prompt(model, tokenizer, reverse_prompt)
            score_forward = forward[0] - forward[1]
            score_reverse = reverse[1] - reverse[0]
            score = (score_forward + score_reverse) / 2.0
            row = {
                "base_id": base_id,
                "sample_id": sample["id"],
                "side": side,
                "expected_decision": "CALL" if side == "valid" else "DECLINE",
                "forward_logit_a": forward[0],
                "forward_logit_b": forward[1],
                "reverse_logit_a": reverse[0],
                "reverse_logit_b": reverse[1],
                "forward_call_score": score_forward,
                "reverse_call_score": score_reverse,
                "decision_score": score,
                "predicted_decision": "CALL" if score > 0 else "DECLINE" if score < 0 else "TIE",
                "label_a_token_id": forward[2],
                "label_b_token_id": forward[3],
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(pairs) * 2:
                print(f"{args.model_name}: 已完成 {index}/{len(pairs) * 2}", flush=True)
    partial.replace(args.output)
    metadata = {
        "model_name": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "pairs": len(pairs),
        "samples": len(rows),
        "seed": args.seed,
        "gpu_memory_fraction": args.gpu_memory_fraction,
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
