#!/usr/bin/env python3
"""运行合同判断接入原生工具调用的 F3 受控实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


BASE_SYSTEM = (
    "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
    "否则不要调用工具。需要调用时只输出结构化工具调用。"
)
DECISION_SYSTEM = (
    "判断当前提供的工具合同能否完整满足用户请求。"
    "只能输出 <decision>CALL</decision> 或 <decision>DECLINE</decision>，不要输出其他内容。"
)
CONDITIONS = (
    "B_self_decision",
    "B2_self_text_decision",
    "C_oracle",
    "D_late_probe",
    "E_inverted",
)
DECISION_RE = re.compile(r"<decision>\s*(CALL|DECLINE)\s*</decision>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """解析冻结数据、探针预测、模型和输出参数。"""
    parser = argparse.ArgumentParser(description="运行 CCPD-v1.1 F3 判断接入实验")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="base")
    parser.add_argument("--splits", default="dev,test_id,test_ood_family,test_ood_constraint")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-decision-tokens", type=int, default=16)
    parser.add_argument("--max-action-tokens", type=int, default=96)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.10)
    parser.add_argument("--resume-partial", action="store_true")
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


def parse_decision(text: str) -> str:
    """严格解析自主判断；格式不合法时返回 INVALID。"""
    match = DECISION_RE.search(text)
    return match.group(1).upper() if match else "INVALID"


def guidance(decision: str) -> str:
    """把冻结的二元判断转换为统一行动指令。"""
    if decision == "CALL":
        return (
            "前置合同判断：FEASIBLE。当前合同能够满足请求；请据此生成且仅生成合法的结构化工具调用。"
        )
    if decision == "DECLINE":
        return (
            "前置合同判断：NOT FEASIBLE。当前合同不能满足请求；请据此拒绝，"
            "不要生成工具调用，只输出：无法使用当前工具完成请求。"
        )
    return "前置合同判断：UNKNOWN。判断格式无效；请严格根据当前合同自行决定是否调用。"


def generate_one(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    """执行一次确定性生成并返回文本、Token 数和延迟。"""
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
    latency = time.perf_counter() - started
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return (
        tokenizer.decode(new_tokens, skip_special_tokens=False).strip(),
        int(inputs["input_ids"].shape[1]),
        int(new_tokens.shape[0]),
        latency,
    )


def expected_decision(side: str) -> str:
    """根据冻结配对侧返回合同可满足性标签。"""
    if side == "plus":
        return "CALL"
    if side == "minus":
        return "DECLINE"
    raise ValueError(f"未知配对侧：{side}")


def text_decision_messages(side: dict[str, Any]) -> list[dict[str, str]]:
    """把同一工具合同作为普通文本展示，隔离工具注册对首个动作的强制偏置。"""
    user_messages = [message["content"] for message in side["messages"] if message.get("role") == "user"]
    if len(user_messages) != 1 or not isinstance(user_messages[0], str):
        raise ValueError("F3 文本判断仅支持恰好一个用户请求")
    contract = json.dumps(side["tools"], ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": DECISION_SYSTEM},
        {
            "role": "user",
            "content": f"用户请求：\n{user_messages[0]}\n\n当前工具合同：\n{contract}",
        },
    ]


def main() -> None:
    """生成自主、Oracle、探针与反转判断接入后的完整原生响应。"""
    args = parse_args()
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"未知 F3 条件：{sorted(unknown)}")
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    pairs = [row for row in read_jsonl(args.pairs.resolve()) if row.get("split") in splits]
    if not pairs:
        raise ValueError("选定划分没有样本")

    probe_data = json.loads(args.probe_summary.resolve().read_text(encoding="utf-8"))
    probe_rows = probe_data.get("late_predictions")
    if not isinstance(probe_rows, list):
        raise ValueError("F2 汇总缺少 late_predictions")
    probe_map = {
        (row["pair_id"], row["side"]): "CALL" if row["predicted_feasible"] else "DECLINE"
        for row in probe_rows
    }
    required_keys = {(pair["pair_id"], side) for pair in pairs for side in ("plus", "minus")}
    if not required_keys.issubset(probe_map):
        raise RuntimeError("F2 后层探针预测没有覆盖全部 F3 样本")

    expected_total = len(conditions) * len(required_keys)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有 F3 预测：{args.output}")
    completed_keys: set[tuple[str, str, str]] = set()
    if partial.exists():
        if not args.resume_partial:
            raise FileExistsError(f"存在未完成预测；需显式传入 --resume-partial：{partial}")
        for row in read_jsonl(partial):
            completed_keys.add((row["condition"], row["pair_id"], row["side"]))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.cuda.reset_peak_memory_stats()
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
                    gold_decision = expected_decision(side_name)
                    decision_text = None
                    decision_tokens = 0
                    decision_latency = 0.0
                    if condition in {"B_self_decision", "B2_self_text_decision"}:
                        decision_messages = (
                            [{"role": "system", "content": DECISION_SYSTEM}, *side["messages"]]
                            if condition == "B_self_decision"
                            else text_decision_messages(side)
                        )
                        decision_prompt = tokenizer.apply_chat_template(
                            decision_messages,
                            tools=side["tools"] if condition == "B_self_decision" else None,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        decision_text, _, decision_tokens, decision_latency = generate_one(
                            model, tokenizer, decision_prompt, args.max_decision_tokens
                        )
                        injected_decision = parse_decision(decision_text)
                    elif condition == "C_oracle":
                        injected_decision = gold_decision
                    elif condition == "D_late_probe":
                        injected_decision = probe_map[(pair["pair_id"], side_name)]
                    else:
                        injected_decision = "DECLINE" if gold_decision == "CALL" else "CALL"

                    action_messages = [
                        {"role": "system", "content": BASE_SYSTEM + guidance(injected_decision)},
                        *side["messages"],
                    ]
                    action_prompt = tokenizer.apply_chat_template(
                        action_messages,
                        tools=side["tools"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    response, input_tokens, output_tokens, action_latency = generate_one(
                        model, tokenizer, action_prompt, args.max_action_tokens
                    )
                    row = {
                        "condition": condition,
                        "pair_id": pair["pair_id"],
                        "source_id": pair["source_id"],
                        "split": pair["split"],
                        "constraint_family": pair["constraint_family"],
                        "side": side_name,
                        "expected_decision": gold_decision,
                        "injected_decision": injected_decision,
                        "decision_correct": injected_decision == gold_decision,
                        "decision_response": decision_text,
                        "decision_output_tokens": decision_tokens,
                        "decision_latency_seconds": round(decision_latency, 6),
                        "response": response,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_seconds": round(action_latency + decision_latency, 6),
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
        "pairs": len(pairs),
        "conditions": conditions,
        "samples": expected_total,
        "seed": args.seed,
        "base_system": BASE_SYSTEM,
        "decision_system": DECISION_SYSTEM,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "pairs_sha256": sha256_file(args.pairs.resolve()),
        "probe_summary_sha256": sha256_file(args.probe_summary.resolve()),
        "predictions_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
