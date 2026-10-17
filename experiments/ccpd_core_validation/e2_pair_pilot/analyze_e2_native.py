#!/usr/bin/env python3
"""分析 E2 Pilot 的 Base/FDSRC 原生调用错误分布。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析项目、数据、预测与输出路径。"""
    parser = argparse.ArgumentParser(description="分析 CCPD E2 原生错误")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
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


def decision(text: str, parse_tool_calls: Any) -> str:
    """严格解析调用、拒绝与非法输出。"""
    parsed = parse_tool_calls(text)
    if not parsed.valid:
        return "INVALID"
    return "CALL" if parsed.calls else "DECLINE"


def main() -> None:
    """汇总每个模型的成对决策和错误分布。"""
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from evaluation.metrics import parse_tool_calls

    pairs = read_jsonl(args.pairs.resolve())
    expected_keys = {(pair["pair_id"], side) for pair in pairs for side in ("plus", "minus")}
    result: dict[str, Any] = {}
    for prediction_path in args.predictions:
        rows = read_jsonl(prediction_path.resolve())
        row_map = {(row["pair_id"], row["side"]): row for row in rows}
        if set(row_map) != expected_keys or len(row_map) != len(rows):
            raise RuntimeError(f"{prediction_path} 的预测键不完整或重复")
        model_name = prediction_path.stem.removesuffix("_native")
        cases: list[dict[str, Any]] = []
        for pair in pairs:
            plus = decision(row_map[(pair["pair_id"], "plus")]["response"], parse_tool_calls)
            minus = decision(row_map[(pair["pair_id"], "minus")]["response"], parse_tool_calls)
            cases.append(
                {
                    "pair_id": pair["pair_id"],
                    "split": pair["split"],
                    "constraint_family": pair["constraint_family"],
                    "plus_decision": plus,
                    "minus_decision": minus,
                    "pair_success": plus == "CALL" and minus == "DECLINE",
                }
            )
        counts = Counter((case["plus_decision"], case["minus_decision"]) for case in cases)
        result[model_name] = {
            "pairs": len(cases),
            "pair_success": sum(case["pair_success"] for case in cases) / len(cases),
            "valid_call_rate": sum(case["plus_decision"] == "CALL" for case in cases) / len(cases),
            "false_reject_rate": sum(case["plus_decision"] == "DECLINE" for case in cases) / len(cases),
            "invalid_call_rate": sum(case["minus_decision"] == "CALL" for case in cases) / len(cases),
            "plus_invalid_output_rate": sum(case["plus_decision"] == "INVALID" for case in cases) / len(cases),
            "minus_invalid_output_rate": sum(case["minus_decision"] == "INVALID" for case in cases) / len(cases),
            "decision_pair_counts": {f"{left}/{right}": count for (left, right), count in sorted(counts.items())},
            "cases": cases,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有 E2 汇总：{args.output}")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
