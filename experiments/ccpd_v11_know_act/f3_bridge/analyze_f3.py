#!/usr/bin/env python3
"""严格汇总 F3 判断到原生行动的转换效果。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析项目、冻结数据、预测和输出路径。"""
    parser = argparse.ArgumentParser(description="分析 CCPD-v1.1 F3 判断接入实验")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL。"""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """按完整配对计算判断、行动、调用内容和成本指标。"""
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        groups[case["pair_id"]][case["side"]] = case
    if not groups or any(set(group) != {"plus", "minus"} for group in groups.values()):
        raise RuntimeError("F3 配对缺失或不完整")
    pairs = list(groups.values())
    decision_correct_cases = [case for case in cases if case["decision_correct"]]
    return {
        "pairs": len(pairs),
        "decision_accuracy": statistics.fmean(case["decision_correct"] for case in cases),
        "decision_pair_success": statistics.fmean(
            group["plus"]["decision_correct"] and group["minus"]["decision_correct"]
            for group in pairs
        ),
        "native_pair_success": statistics.fmean(
            group["plus"]["decision"] == "CALL" and group["minus"]["decision"] == "DECLINE"
            for group in pairs
        ),
        "complete_pair_success": statistics.fmean(
            group["plus"]["complete_call_correct"] and group["minus"]["decision"] == "DECLINE"
            for group in pairs
        ),
        "valid_call_rate": statistics.fmean(group["plus"]["decision"] == "CALL" for group in pairs),
        "false_call_rate": statistics.fmean(group["minus"]["decision"] == "CALL" for group in pairs),
        "false_reject_rate": statistics.fmean(group["plus"]["decision"] == "DECLINE" for group in pairs),
        "valid_complete_call_rate": statistics.fmean(group["plus"]["complete_call_correct"] for group in pairs),
        "action_correct_given_decision_correct": (
            statistics.fmean(
                (case["side"] == "plus" and case["decision"] == "CALL")
                or (case["side"] == "minus" and case["decision"] == "DECLINE")
                for case in decision_correct_cases
            )
            if decision_correct_cases else None
        ),
        "invalid_output_rate": statistics.fmean(case["decision"] == "INVALID" for case in cases),
        "mean_total_latency_seconds": statistics.fmean(case["latency_seconds"] for case in cases),
        "mean_decision_tokens": statistics.fmean(case["decision_output_tokens"] for case in cases),
        "mean_action_tokens": statistics.fmean(case["output_tokens"] for case in cases),
    }


def main() -> None:
    """复用 F1 严格 Parser 后，按条件和划分输出 F3 指标。"""
    args = parse_args()
    sys.path.insert(0, str(args.project_root.resolve()))
    from evaluation.metrics import parse_tool_calls
    from experiments.ccpd_v11_know_act.f1_prompt.analyze_f1 import call_exact, classify_response

    pair_map = {row["pair_id"]: row for row in read_jsonl(args.pairs.resolve())}
    predictions = read_jsonl(args.predictions.resolve())
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for prediction in predictions:
        key = (prediction["condition"], prediction["pair_id"], prediction["side"])
        if key in seen:
            raise RuntimeError(f"预测键重复：{key}")
        seen.add(key)
        pair = pair_map[prediction["pair_id"]]
        side = pair[prediction["side"]]
        decision, parse_valid, calls = classify_response(prediction["response"], parse_tool_calls)
        cases.append(
            {
                **prediction,
                "decision": decision,
                "parse_valid": parse_valid,
                "complete_call_correct": decision == "CALL" and call_exact(calls, side["expected_calls"]),
            }
        )

    conditions = sorted({case["condition"] for case in cases})
    splits = sorted({case["split"] for case in cases})
    result = {
        "overall": {
            condition: summarize([case for case in cases if case["condition"] == condition])
            for condition in conditions
        },
        "by_split": {
            condition: {
                split: summarize(
                    [case for case in cases if case["condition"] == condition and case["split"] == split]
                )
                for split in splits
            }
            for condition in conditions
        },
        "cases": cases,
    }
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖 F3 汇总：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "by_split": result["by_split"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
