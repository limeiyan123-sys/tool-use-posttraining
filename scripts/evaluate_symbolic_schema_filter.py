#!/usr/bin/env python3
"""评估无需训练的 JSON Schema 符号过滤基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 Symbolic Schema Filter")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 出现空行")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """创建 JSONL 文件并拒绝覆盖。"""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    """按冻结清单读取并校验工具世界数据。"""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for filename, metadata in manifest["datasets"].items():
        path = dataset_dir / filename
        if sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"数据哈希不匹配：{path}")
        current = read_jsonl(path)
        if len(current) != metadata["samples"]:
            raise RuntimeError(f"数据样本数不匹配：{path}")
        rows.extend(current)
    return rows


def serialize_calls(calls: list[dict[str, Any]]) -> str:
    """把保留调用序列化为项目的严格工具协议。"""
    if not calls:
        return "The current tool environment cannot complete this request."
    return "\n".join(
        f"<tool_call>\n{json.dumps(call, ensure_ascii=False, sort_keys=True)}\n</tool_call>"
        for call in calls
    )


def filter_prediction(
    sample: dict[str, Any], prediction: dict[str, Any], parse_tool_calls: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """删除未声明工具调用及违反当前 JSON Schema 的调用。"""
    from evaluation.metrics import validate_arguments

    parsed = parse_tool_calls(prediction["response"])
    if not parsed.valid:
        return dict(prediction), []
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in sample["tools"]
    }
    kept: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for call in parsed.calls:
        schema = schemas.get(call["name"])
        if schema is None:
            blocked.append({"name": call["name"], "reason": "unknown_tool"})
            continue
        errors = validate_arguments(call["arguments"], schema)
        if errors:
            blocked.append(
                {
                    "name": call["name"],
                    "reason": "schema_violation",
                    "messages": errors,
                }
            )
            continue
        kept.append(call)
    filtered = dict(prediction)
    filtered["response"] = serialize_calls(kept)
    filtered["symbolic_filter"] = {
        "input_calls": len(parsed.calls),
        "kept_calls": len(kept),
        "blocked_calls": blocked,
    }
    return filtered, blocked


def compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """提取总体与关键工具世界的 Decision Exact。"""
    return {
        "overall": summary["overall"]["decision_exact"],
        "capability_substitution": summary["by_world"]["capability_substitution"]["decision_exact"],
        "distractor": summary["by_world"]["distractor"]["decision_exact"],
        "original": summary["by_world"]["original"]["decision_exact"],
    }


def percent(value: float) -> str:
    """将比例格式化为百分比。"""
    return f"{value * 100:.2f}%"


def render_report(run_name: str, result: dict[str, Any]) -> str:
    """渲染符号过滤基线报告。"""
    original = result["original_metrics"]
    filtered = result["filtered_metrics"]
    lines = [
        f"# {run_name} Symbolic Schema Filter",
        "",
        "## Decision Exact",
        "",
        "| 场景 | 原始 | 过滤后 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for world in ("overall", "capability_substitution", "distractor", "original"):
        delta = filtered[world] - original[world]
        lines.append(
            f"| {world} | {percent(original[world])} | {percent(filtered[world])} | {delta * 100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 过滤统计",
            "",
            f"- 被过滤调用数：{result['blocked_calls']}",
            f"- 受影响样本数：{result['affected_samples']}",
            f"- 错误转正确：{result['wrong_to_correct']}",
            f"- 正确转错误：{result['correct_to_wrong']}",
            f"- 原因：{json.dumps(result['blocked_reason_counts'], ensure_ascii=False, sort_keys=True)}",
            "",
            "## 解释边界",
            "",
            "- 该基线只验证模型已经生成的参数是否满足当前 JSON Schema，不生成或猜测缺失需求。",
            "- 用户要求的值若对应 Schema 中的可选字段，而模型省略该字段，纯 Schema 校验无法发现。",
            "- 过滤后不自动重试模型，因此它衡量确定性拒绝能够直接覆盖多少错误，而不是完整 Agent 上限。",
            "- 非法输出保持原样，不使用宽松解析或静默修复。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """运行符号过滤，比较过滤前后的真实评测结果。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.metrics import parse_tool_calls
    from evaluation.tool_world_metrics import aggregate_tool_world, evaluate_tool_world_case

    dataset_dir = args.dataset_dir.resolve()
    samples = load_dataset(dataset_dir)
    predictions = read_jsonl(args.predictions.resolve())
    prediction_map = {row["id"]: row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise RuntimeError("预测中存在重复 ID")
    if {sample["id"] for sample in samples} != set(prediction_map):
        raise RuntimeError("预测 ID 与数据集不一致")

    filtered_predictions: list[dict[str, Any]] = []
    blocked_by_id: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        filtered, blocked = filter_prediction(sample, prediction_map[sample["id"]], parse_tool_calls)
        filtered_predictions.append(filtered)
        blocked_by_id[sample["id"]] = blocked

    original_cases = [
        evaluate_tool_world_case(sample, prediction_map[sample["id"]]) for sample in samples
    ]
    filtered_map = {row["id"]: row for row in filtered_predictions}
    filtered_cases = [
        evaluate_tool_world_case(sample, filtered_map[sample["id"]]) for sample in samples
    ]
    original_summary = aggregate_tool_world(original_cases)
    filtered_summary = aggregate_tool_world(filtered_cases)
    paired_cases: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for original, filtered in zip(original_cases, filtered_cases):
        blocked = blocked_by_id[original["id"]]
        for item in blocked:
            reason_counts[item["reason"]] += 1
        paired_cases.append(
            {
                "id": original["id"],
                "world_type": original["world_type"],
                "original_decision_exact": original["decision_exact"],
                "filtered_decision_exact": filtered["decision_exact"],
                "original_error_type": original["error_type"],
                "filtered_error_type": filtered["error_type"],
                "blocked_calls": blocked,
            }
        )
    result = {
        "run_name": args.run_name,
        "samples": len(samples),
        "original_metrics": compact_metrics(original_summary),
        "filtered_metrics": compact_metrics(filtered_summary),
        "blocked_calls": sum(len(items) for items in blocked_by_id.values()),
        "affected_samples": sum(bool(items) for items in blocked_by_id.values()),
        "wrong_to_correct": sum(
            not row["original_decision_exact"] and row["filtered_decision_exact"] for row in paired_cases
        ),
        "correct_to_wrong": sum(
            row["original_decision_exact"] and not row["filtered_decision_exact"] for row in paired_cases
        ),
        "blocked_reason_counts": dict(sorted(reason_counts.items())),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / f"{args.run_name}_filtered_predictions.jsonl"
    cases_path = output_dir / f"{args.run_name}_paired_cases.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    report_path = output_dir / f"{args.run_name}_report.md"
    for path in (filtered_path, cases_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    write_jsonl(filtered_path, filtered_predictions)
    write_jsonl(cases_path, paired_cases)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(args.run_name, result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
