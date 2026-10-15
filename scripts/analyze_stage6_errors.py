#!/usr/bin/env python3
"""拆分 Stage 6 动态工具环境中的决策、工具选择与参数错误。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


METHOD_PATHS = {
    "FDSRC": Path("reports/tool_world_final/fdsrc_evaluation/fdsrc_case_results.jsonl"),
    "FWCT": Path(
        "reports/tool_world_final/failure_weighted_evaluation/"
        "failure_weighted_case_results.jsonl"
    ),
    "FCTW": Path(
        "reports/stage5_fctw/tool_world_final/fctw_evaluation/"
        "fctw_case_results.jsonl"
    ),
}

TARGET_WORLDS = ("overall", "capability_substitution", "distractor", "original")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="拆分 Stage 6 工具调用错误")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL，拒绝空行和非对象记录。"""
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


def classify_error(row: dict[str, Any]) -> str:
    """按统一优先级将每条预测映射为互斥错误类型。"""
    if not row.get("parse_valid", False):
        return "invalid_output"
    gold_call = bool(row.get("expected_calls"))
    predicted_call = bool(row.get("predicted_calls"))
    if not gold_call and predicted_call:
        return "false_call"
    if gold_call and not predicted_call:
        return "false_reject"
    if gold_call and predicted_call and not row.get("tool_correct", False):
        return "wrong_tool"
    if gold_call and predicted_call and not row.get("argument_exact", False):
        return "argument_failure"
    if row.get("decision_exact", False):
        return "correct"
    return "schema_or_protocol_failure"


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """计算比例；分母为零时显式返回空值。"""
    return numerator / denominator if denominator else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """计算动作、工具、参数和端到端四层指标。"""
    error_counts = Counter(classify_error(row) for row in rows)
    action_correct = sum(
        bool(row.get("parse_valid", False))
        and bool(row.get("expected_calls")) == bool(row.get("predicted_calls"))
        for row in rows
    )
    should_call = [row for row in rows if row.get("expected_calls")]
    correct_tool = [
        row
        for row in should_call
        if row.get("parse_valid", False)
        and row.get("predicted_calls")
        and row.get("tool_correct", False)
    ]
    argument_exact = sum(bool(row.get("argument_exact", False)) for row in correct_tool)
    decision_exact = sum(bool(row.get("decision_exact", False)) for row in rows)
    return {
        "samples": len(rows),
        "should_call_samples": len(should_call),
        "call_reject_correct": action_correct,
        "call_reject_accuracy": safe_ratio(action_correct, len(rows)),
        "correct_tool": len(correct_tool),
        "tool_selection_accuracy_on_should_call": safe_ratio(len(correct_tool), len(should_call)),
        "argument_exact_on_correct_tool": argument_exact,
        "argument_exact_accuracy_on_correct_tool": safe_ratio(argument_exact, len(correct_tool)),
        "decision_exact": decision_exact,
        "decision_exact_accuracy": safe_ratio(decision_exact, len(rows)),
        "error_counts": dict(sorted(error_counts.items())),
    }


def percent(value: float | None) -> str:
    """将可空比例格式化为百分比。"""
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_report(result: dict[str, Any]) -> str:
    """渲染简体中文诊断报告。"""
    lines = [
        "# Stage 6 动态工具调用分层错误诊断",
        "",
        "## 指标定义",
        "",
        "- Call/Reject：严格解析成功，且是否调用与 Gold 一致。",
        "- Tool Selection：以全部应调用样本为分母，错误拒绝和非法输出计错。",
        "- Argument Exact：仅在严格解析且工具选择正确的样本上计算条件准确率。",
        "- End-to-end：沿用既有 Decision Exact，不用条件指标替代最终表现。",
        "",
        "## 核心结果",
        "",
        "| 方法 | 场景 | N | Call/Reject | Tool@ShouldCall | ArgExact@CorrectTool | End-to-end |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method, worlds in result["methods"].items():
        for world in TARGET_WORLDS:
            metrics = worlds[world]
            lines.append(
                f"| {method} | {world} | {metrics['samples']} | "
                f"{percent(metrics['call_reject_accuracy'])} | "
                f"{percent(metrics['tool_selection_accuracy_on_should_call'])} | "
                f"{percent(metrics['argument_exact_accuracy_on_correct_tool'])} | "
                f"{percent(metrics['decision_exact_accuracy'])} |"
            )

    lines.extend(["", "## Substitution 错误计数", ""])
    lines.append("| 方法 | Correct | False Call | False Reject | Wrong Tool | Arg Failure | Invalid |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method, worlds in result["methods"].items():
        counts = worlds["capability_substitution"]["error_counts"]
        lines.append(
            f"| {method} | {counts.get('correct', 0)} | {counts.get('false_call', 0)} | "
            f"{counts.get('false_reject', 0)} | {counts.get('wrong_tool', 0)} | "
            f"{counts.get('argument_failure', 0)} | {counts.get('invalid_output', 0)} |"
        )

    lines.extend(["", "## Distractor 错误计数", ""])
    lines.append("| 方法 | Correct | False Call | False Reject | Wrong Tool | Arg Failure | Invalid |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method, worlds in result["methods"].items():
        counts = worlds["distractor"]["error_counts"]
        lines.append(
            f"| {method} | {counts.get('correct', 0)} | {counts.get('false_call', 0)} | "
            f"{counts.get('false_reject', 0)} | {counts.get('wrong_tool', 0)} | "
            f"{counts.get('argument_failure', 0)} | {counts.get('invalid_output', 0)} |"
        )

    lines.extend(
        [
            "",
            "## 研究判断",
            "",
            "1. FCTW 的 Substitution 仅 2/60 端到端正确，主要错误是 44 次 False Call；"
            "另有 14 个仍需保留部分合法调用的样本发生 Wrong Tool。参数生成不是该场景的首要瓶颈。",
            "2. FWCT 将 Substitution 的正确拒绝显著提高到 34/60，但仍有 12 次 False Call、"
            "13 次 Wrong Tool，说明它学到了一部分能力约束信号，但不稳定。",
            "3. FWCT 的 Distractor 退化直接来自 14 次 False Reject；FCTW 消除了这批错误拒绝，"
            "却几乎完全丢失 FWCT 的 Substitution 收益。",
            "4. FCTW 与 FDSRC 在 Original 和 Distractor 的分层指标接近，说明核心冲突位于"
            "“语义相关但能力不兼容”的判断，而非一般工具调用格式或普通参数生成。",
            "5. 下一步应使用单工具 Probe 区分：模型是不会判断单个 Schema 的兼容性，"
            "还是能够判断但无法把判断整合进多工具行动。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """读取三组既有预测，保存统一分层统计和报告。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage6_error_decomposition.json"
    report_path = output_dir / "capability_diagnosis_a1.md"
    for path in (json_path, report_path):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")

    result: dict[str, Any] = {"methods": {}}
    expected_ids: set[str] | None = None
    for method, relative_path in METHOD_PATHS.items():
        rows = read_jsonl(project_root / relative_path)
        ids = {str(row["id"]) for row in rows}
        if len(ids) != len(rows):
            raise RuntimeError(f"{method} 存在重复样本 ID")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise RuntimeError(f"{method} 与其他方法的评测样本不一致")
        result["methods"][method] = {}
        for world in TARGET_WORLDS:
            selected = rows if world == "overall" else [row for row in rows if row["world_type"] == world]
            result["methods"][method][world] = summarize(selected)

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
