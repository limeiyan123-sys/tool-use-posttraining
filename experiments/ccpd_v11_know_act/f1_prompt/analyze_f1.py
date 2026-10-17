#!/usr/bin/env python3
"""严格分析 F1 原生提示条件下的决策、调用内容与成本。"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")
REJECTION_PATTERN = re.compile(
    r"(?:无法|不能|拒绝|不可以|不可用|不能使用|cannot|can't|unable|decline|not\s+possible|"
    r"no\s+(?:available|suitable|compatible)\s+tool)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    """解析项目、冻结数据、预测与输出路径。"""
    parser = argparse.ArgumentParser(description="分析 CCPD-v1.1 F1 提示控制")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def clean_text(text: str) -> str:
    """仅移除已知生成终止 Token，不修复正文。"""
    result = text
    for token in SPECIAL_TOKENS:
        result = result.replace(token, "")
    return result.strip()


def values_equal(left: Any, right: Any) -> bool:
    """比较 JSON 参数值，数值允许 int/float 等值表示。"""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return type(left) is type(right) and left == right


def call_exact(calls: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    """验证单调用工具名、参数集合和值均精确匹配。"""
    if len(calls) != 1 or len(expected) != 1:
        return False
    predicted = calls[0]
    gold = expected[0]
    if predicted.get("name") != gold.get("name"):
        return False
    arguments = predicted.get("arguments")
    gold_arguments = gold.get("arguments")
    if not isinstance(arguments, dict) or not isinstance(gold_arguments, dict):
        return False
    return set(arguments) == set(gold_arguments) and all(
        values_equal(arguments[name], gold_arguments[name]) for name in gold_arguments
    )


def classify_response(text: str, parse_tool_calls: Any) -> tuple[str, bool, list[dict[str, Any]]]:
    """严格区分工具调用、明确拒绝和无效无调用文本。"""
    parsed = parse_tool_calls(text)
    if not parsed.valid:
        return "INVALID", False, []
    if parsed.calls:
        return "CALL", True, parsed.calls
    cleaned = clean_text(text)
    if cleaned and REJECTION_PATTERN.search(cleaned):
        return "DECLINE", True, []
    return "INVALID", False, []


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """按配对汇总原生决策、完整调用和推理成本。"""
    if not cases:
        raise ValueError("没有可汇总的 F1 样本")
    pair_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        pair_groups[case["pair_id"]][case["side"]] = case
    if any(set(group) != {"plus", "minus"} for group in pair_groups.values()):
        raise RuntimeError("F1 配对两侧不完整")
    pairs = list(pair_groups.values())
    return {
        "pairs": len(pairs),
        "native_pair_success": statistics.fmean(
            group["plus"]["decision"] == "CALL" and group["minus"]["decision"] == "DECLINE"
            for group in pairs
        ),
        "complete_pair_success": statistics.fmean(
            group["plus"]["complete_call_correct"] and group["minus"]["decision"] == "DECLINE"
            for group in pairs
        ),
        "valid_call_rate": statistics.fmean(group["plus"]["decision"] == "CALL" for group in pairs),
        "invalid_call_rate": statistics.fmean(group["minus"]["decision"] == "CALL" for group in pairs),
        "false_reject_rate": statistics.fmean(group["plus"]["decision"] == "DECLINE" for group in pairs),
        "false_call_rate": statistics.fmean(group["minus"]["decision"] == "CALL" for group in pairs),
        "plus_invalid_output_rate": statistics.fmean(group["plus"]["decision"] == "INVALID" for group in pairs),
        "minus_invalid_output_rate": statistics.fmean(group["minus"]["decision"] == "INVALID" for group in pairs),
        "valid_complete_call_rate": statistics.fmean(group["plus"]["complete_call_correct"] for group in pairs),
        "mean_input_tokens": statistics.fmean(case["input_tokens"] for case in cases),
        "mean_output_tokens": statistics.fmean(case["output_tokens"] for case in cases),
        "mean_latency_seconds": statistics.fmean(case["latency_seconds"] for case in cases),
        "fixed_rejection_format_rate": statistics.fmean(case["fixed_rejection_format"] for case in cases),
    }


def main() -> None:
    """把逐样本生成映射到严格行为指标并分条件、分划分汇总。"""
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from evaluation.metrics import parse_tool_calls

    pairs = read_jsonl(args.pairs.resolve())
    pair_map = {row["pair_id"]: row for row in pairs}
    predictions = read_jsonl(args.predictions.resolve())
    seen_keys: set[tuple[str, str, str]] = set()
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        key = (prediction["condition"], prediction["pair_id"], prediction["side"])
        if key in seen_keys:
            raise RuntimeError(f"预测键重复：{key}")
        seen_keys.add(key)
        pair = pair_map.get(prediction["pair_id"])
        if pair is None:
            raise KeyError(f"预测引用未知配对：{prediction['pair_id']}")
        side = pair[prediction["side"]]
        decision, parse_valid, calls = classify_response(prediction["response"], parse_tool_calls)
        cleaned = clean_text(prediction["response"])
        cases.append(
            {
                **prediction,
                "decision": decision,
                "parse_valid": parse_valid,
                "complete_call_correct": (
                    decision == "CALL" and call_exact(calls, side["expected_calls"])
                ),
                "fixed_rejection_format": cleaned == "无法使用当前工具完成请求。",
            }
        )

    result: dict[str, Any] = {"overall": {}, "by_split": {}, "cases": cases}
    conditions = sorted({case["condition"] for case in cases})
    splits = sorted({case["split"] for case in cases})
    for condition in conditions:
        selected = [case for case in cases if case["condition"] == condition]
        result["overall"][condition] = summarize(selected)
        result["by_split"][condition] = {
            split: summarize([case for case in selected if case["split"] == split])
            for split in splits
            if any(case["split"] == split for case in selected)
        }
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖 F1 汇总：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "by_split": result["by_split"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
