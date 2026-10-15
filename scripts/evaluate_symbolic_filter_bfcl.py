#!/usr/bin/env python3
"""在冻结 BFCL holdout 上检查符号过滤是否造成普通工具调用退化。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluate_symbolic_schema_filter import (
    filter_prediction,
    load_dataset,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估符号过滤的 BFCL 回归")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def select_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """提取 BFCL 回归所需的三个核心指标。"""
    overall = summary["overall"]
    return {
        "tool_accuracy": overall["tool_accuracy"],
        "argument_accuracy": overall["argument_accuracy"],
        "argument_exact": overall["argument_exact_match"],
    }


def percent(value: float) -> str:
    """将比例格式化为百分比。"""
    return f"{value * 100:.2f}%"


def render_report(run_name: str, result: dict[str, Any]) -> str:
    """渲染 BFCL 回归报告。"""
    original = result["original"]
    filtered = result["filtered"]
    lines = [
        f"# {run_name} Symbolic Filter BFCL Regression",
        "",
        "| 指标 | 原始 | 过滤后 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("tool_accuracy", "argument_accuracy", "argument_exact"):
        delta = filtered[key] - original[key]
        lines.append(
            f"| {key} | {percent(original[key])} | {percent(filtered[key])} | {delta * 100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"- 过滤调用数：{result['blocked_calls']}",
            f"- 正确转错误：{result['correct_to_wrong']}",
            "- 非法输出保持不变，未进行宽松解析。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """过滤预测并比较 BFCL 指标。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.metrics import aggregate, evaluate_case, parse_tool_calls

    samples = load_dataset(args.dataset_dir.resolve())
    predictions = read_jsonl(args.predictions.resolve())
    prediction_map = {row["id"]: row for row in predictions}
    if {sample["id"] for sample in samples} != set(prediction_map):
        raise RuntimeError("预测 ID 与 BFCL holdout 不一致")
    filtered_predictions: list[dict[str, Any]] = []
    blocked_calls = 0
    for sample in samples:
        filtered, blocked = filter_prediction(sample, prediction_map[sample["id"]], parse_tool_calls)
        filtered_predictions.append(filtered)
        blocked_calls += len(blocked)
    filtered_map = {row["id"]: row for row in filtered_predictions}
    original_cases = [evaluate_case(sample, prediction_map[sample["id"]]) for sample in samples]
    filtered_cases = [evaluate_case(sample, filtered_map[sample["id"]]) for sample in samples]
    result = {
        "run_name": args.run_name,
        "samples": len(samples),
        "original": select_metrics(aggregate(original_cases)),
        "filtered": select_metrics(aggregate(filtered_cases)),
        "blocked_calls": blocked_calls,
        "correct_to_wrong": sum(
            original["error_type"] == "none" and filtered["error_type"] != "none"
            for original, filtered in zip(original_cases, filtered_cases)
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.run_name}_filtered_predictions.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    report_path = output_dir / f"{args.run_name}_report.md"
    for path in (predictions_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    write_jsonl(predictions_path, filtered_predictions)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(args.run_name, result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
