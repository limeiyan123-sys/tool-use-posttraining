#!/usr/bin/env python3
"""在训练前采样少量配对 Rollout，测量 B1/B2/B3 的真实奖励稀疏性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析模型、数据、采样和输出参数。"""
    parser = argparse.ArgumentParser(description="CCPD E3 训练前奖励信号检查")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=8)
    parser.add_argument("--rollouts-per-side", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261017)
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
    """构造冻结的原生工具调用消息。"""
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
                "否则不要调用工具。需要调用时只输出结构化工具调用。"
            ),
        },
        *side["messages"],
    ]


def schema_accepts(arguments: dict[str, Any], tool: dict[str, Any]) -> bool:
    """验证 Pilot 实际使用的 required、enum 与 maximum。"""
    parameters = tool["function"]["parameters"]
    if any(name not in arguments for name in parameters.get("required", [])):
        return False
    for name, value in arguments.items():
        schema = parameters["properties"].get(name)
        if schema is None:
            return False
        if "enum" in schema and value not in schema["enum"]:
            return False
        if "maximum" in schema:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value > schema["maximum"]:
                return False
    return True


def canonical_call(call: dict[str, Any]) -> str:
    """把调用规范化为可比较字符串。"""
    return json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evaluate_response(
    response: str, side_name: str, side: dict[str, Any], parse_tool_calls: Any
) -> dict[str, bool]:
    """同时计算决策正确与严格内容正确。"""
    parsed = parse_tool_calls(response)
    if not parsed.valid:
        return {"format_valid": False, "decision_correct": False, "content_correct": False}
    if side_name == "minus":
        declined = len(parsed.calls) == 0
        return {"format_valid": True, "decision_correct": declined, "content_correct": declined}
    called = len(parsed.calls) > 0
    tool = side["tools"][0]
    schema_valid = called and all(
        call["name"] == tool["function"]["name"] and schema_accepts(call["arguments"], tool)
        for call in parsed.calls
    )
    exact = sorted(canonical_call(call) for call in parsed.calls) == sorted(
        canonical_call(call) for call in side["expected_calls"]
    )
    return {
        "format_valid": True,
        "decision_correct": called,
        "content_correct": bool(schema_valid and exact),
    }


def standardized_advantages(rewards: list[float]) -> list[float]:
    """按 GRPO 常见的组内总体标准差计算优势。"""
    mean = statistics.fmean(rewards)
    variance = statistics.fmean((reward - mean) ** 2 for reward in rewards)
    if variance == 0:
        return [0.0 for _ in rewards]
    scale = math.sqrt(variance)
    return [(reward - mean) / scale for reward in rewards]


def main() -> None:
    """采样并精确统计三种分组/奖励方案的有效优势。"""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有预检结果：{args.output}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = args.project_root.resolve()
    import sys

    sys.path.insert(0, str(root))
    from evaluation.metrics import parse_tool_calls

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.cuda.reset_peak_memory_stats()
    pairs = [pair for pair in read_jsonl(args.pairs.resolve()) if pair["split"] == "train"]
    pairs = pairs[: args.pair_count]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path.resolve(), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path.resolve(),
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    started = time.perf_counter()
    pair_records: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs, 1):
        side_evaluations: dict[str, list[dict[str, Any]]] = {}
        for side_name in ("plus", "minus"):
            side = pair[side_name]
            prompt = tokenizer.apply_chat_template(
                direct_messages(side), tools=side["tools"], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.rollouts_per_side,
                    use_cache=True,
                )
            evaluations: list[dict[str, Any]] = []
            for sequence in generated:
                new_tokens = sequence[inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
                evaluations.append(
                    {"response": response, **evaluate_response(response, side_name, side, parse_tool_calls)}
                )
            side_evaluations[side_name] = evaluations
        plus = side_evaluations["plus"]
        minus = side_evaluations["minus"]
        independent_rewards = [float(row["content_correct"]) for row in plus + minus]
        pair_rewards = [
            float(plus[index]["content_correct"] and minus[index]["content_correct"])
            for index in range(args.rollouts_per_side)
        ]
        copied_pair_rewards = [value for reward in pair_rewards for value in (reward, reward)]
        plus_rewards = [float(row["content_correct"]) for row in plus]
        minus_rewards = [float(row["content_correct"]) for row in minus]
        pair_records.append(
            {
                "pair_id": pair["pair_id"],
                "plus": plus,
                "minus": minus,
                "b1_plus_zero_variance": not any(standardized_advantages(plus_rewards)),
                "b1_minus_zero_variance": not any(standardized_advantages(minus_rewards)),
                "b2_mixed_zero_variance": not any(standardized_advantages(independent_rewards)),
                "b3_paired_zero_variance": not any(standardized_advantages(copied_pair_rewards)),
                "b2_effective_advantages": sum(abs(value) > 0 for value in standardized_advantages(independent_rewards)),
                "b3_effective_advantages": sum(abs(value) > 0 for value in standardized_advantages(copied_pair_rewards)),
                "pair_rewards": pair_rewards,
            }
        )
        print(f"预检已完成 {pair_index}/{len(pairs)}", flush=True)
    plus_rows = [row for pair in pair_records for row in pair["plus"]]
    minus_rows = [row for pair in pair_records for row in pair["minus"]]
    summary = {
        "pairs": len(pair_records),
        "rollouts_per_side": args.rollouts_per_side,
        "plus_decision_success_rate": statistics.fmean(row["decision_correct"] for row in plus_rows),
        "minus_decision_success_rate": statistics.fmean(row["decision_correct"] for row in minus_rows),
        "plus_content_success_rate": statistics.fmean(row["content_correct"] for row in plus_rows),
        "minus_content_success_rate": statistics.fmean(row["content_correct"] for row in minus_rows),
        "joint_pair_success_rate": statistics.fmean(
            reward for pair in pair_records for reward in pair["pair_rewards"]
        ),
        "b1_plus_zero_variance_rate": statistics.fmean(
            pair["b1_plus_zero_variance"] for pair in pair_records
        ),
        "b1_minus_zero_variance_rate": statistics.fmean(
            pair["b1_minus_zero_variance"] for pair in pair_records
        ),
        "b2_mixed_zero_variance_rate": statistics.fmean(
            pair["b2_mixed_zero_variance"] for pair in pair_records
        ),
        "b3_paired_zero_variance_rate": statistics.fmean(
            pair["b3_paired_zero_variance"] for pair in pair_records
        ),
        "b2_effective_advantage_fraction": sum(
            pair["b2_effective_advantages"] for pair in pair_records
        ) / (len(pair_records) * args.rollouts_per_side * 2),
        "b3_effective_advantage_fraction": sum(
            pair["b3_effective_advantages"] for pair in pair_records
        ) / (len(pair_records) * args.rollouts_per_side * 2),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0.0
        ),
        "pairs_sha256": sha256_file(args.pairs.resolve()),
    }
    payload = {"config": vars(args) | {"project_root": str(root), "pairs": str(args.pairs), "model_path": str(args.model_path), "output": str(args.output)}, "summary": summary, "pair_records": pair_records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
