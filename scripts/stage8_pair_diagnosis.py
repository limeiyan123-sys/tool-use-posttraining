#!/usr/bin/env python3
"""利用冻结的 Original/Ephemeral 配对结果定位接口表面变化造成的错误。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析输入输出路径。"""
    parser = argparse.ArgumentParser(description="Stage 8 接口表面配对诊断")
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL 文件。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 出现空行")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 不是对象")
            rows.append(value)
    return rows


def interface_invariants(original: dict[str, Any], ephemeral: dict[str, Any]) -> list[str]:
    """验证配对世界除接口标识和顺序外保持语义描述与 Schema 不变。"""
    errors: list[str] = []
    if original["messages"] != ephemeral["messages"]:
        errors.append("query_changed")
    if len(original["tools"]) != len(ephemeral["tools"]):
        errors.append("tool_count_changed")
        return errors
    original_by_name = {tool["function"]["name"]: tool["function"] for tool in original["tools"]}
    for new_name, mapping in ephemeral["semantic_map"].items():
        old_name = mapping["canonical_name"]
        if old_name not in original_by_name:
            errors.append(f"missing_original_tool:{old_name}")
            continue
        new_tool = next(
            (tool["function"] for tool in ephemeral["tools"] if tool["function"]["name"] == new_name),
            None,
        )
        if new_tool is None:
            errors.append(f"missing_ephemeral_tool:{new_name}")
            continue
        old_tool = original_by_name[old_name]
        if old_tool.get("description") != new_tool.get("description"):
            errors.append(f"tool_description_changed:{old_name}")
        reverse_args = mapping["arguments"]
        old_properties = old_tool["parameters"].get("properties", {})
        new_properties = new_tool["parameters"].get("properties", {})
        for new_arg, old_arg in reverse_args.items():
            if old_properties.get(old_arg) != new_properties.get(new_arg):
                errors.append(f"argument_schema_changed:{old_name}:{old_arg}")
    return errors


def ratio(numerator: int, denominator: int) -> float | None:
    """返回显式分母比例。"""
    return numerator / denominator if denominator else None


def paired_sign_test_p(degradations: int, improvements: int) -> float | None:
    """对不一致配对执行双侧精确二项检验，不把一致配对当作独立证据。"""
    discordant = degradations + improvements
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, index) * (0.5**discordant)
        for index in range(min(degradations, improvements) + 1)
    )
    return min(1.0, 2 * tail)


def main() -> None:
    """计算接口表面变化下的配对转移和错误类型。"""
    args = parse_args()
    attribution = read_jsonl(args.attribution)
    samples = read_jsonl(args.dataset)
    sample_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for sample in samples:
        sample_groups[sample["base_id"]][sample["world_type"]] = sample

    invariant_errors: list[dict[str, Any]] = []
    for base_id, worlds in sample_groups.items():
        errors = interface_invariants(worlds["original"], worlds["ephemeral_interface"])
        if errors:
            invariant_errors.append({"base_id": base_id, "errors": errors})
    if invariant_errors:
        raise RuntimeError(f"接口配对语义不等价：{invariant_errors[:3]}")

    record_groups: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in attribution:
        record_groups[row["method"]][row["base_id"]][row["world_type"]] = row

    summary: dict[str, Any] = {
        "design": {
            "base_tasks": len(sample_groups),
            "paired_worlds": ["original", "ephemeral_interface"],
            "invariant_check_errors": 0,
            "controlled_change": "工具名、参数名与接口/字段顺序；自然语言描述和参数 Schema 语义保持不变。",
        },
        "methods": {},
    }
    pair_rows: list[dict[str, Any]] = []
    for method, bases in sorted(record_groups.items()):
        transitions: Counter[str] = Counter()
        degradation_errors: Counter[str] = Counter()
        improvement_errors: Counter[str] = Counter()
        grouped_transitions: dict[str, Counter[str]] = defaultdict(Counter)
        for base_id, worlds in sorted(bases.items()):
            original = worlds["original"]["filtered"]
            ephemeral = worlds["ephemeral_interface"]["filtered"]
            original_ok = original["decision_exact"]
            ephemeral_ok = ephemeral["decision_exact"]
            transition = {
                (True, True): "both_correct",
                (True, False): "original_correct_ephemeral_wrong",
                (False, True): "original_wrong_ephemeral_correct",
                (False, False): "both_wrong",
            }[(original_ok, ephemeral_ok)]
            transitions[transition] += 1
            tool_group = (
                "single_tool"
                if len(sample_groups[base_id]["original"]["tools"]) == 1
                else "multi_tool"
            )
            grouped_transitions[tool_group][transition] += 1
            if original_ok and not ephemeral_ok:
                degradation_errors[ephemeral["primary_failure"]] += 1
            if not original_ok and ephemeral_ok:
                improvement_errors[original["primary_failure"]] += 1
            pair_rows.append(
                {
                    "method": method,
                    "base_id": base_id,
                    "transition": transition,
                    "tool_group": tool_group,
                    "category": sample_groups[base_id]["original"]["category"],
                    "original": original,
                    "ephemeral": ephemeral,
                }
            )
        total = len(bases)
        original_correct = transitions["both_correct"] + transitions["original_correct_ephemeral_wrong"]
        ephemeral_correct = transitions["both_correct"] + transitions["original_wrong_ephemeral_correct"]
        summary["methods"][method] = {
            "pairs": total,
            "original_correct": original_correct,
            "original_accuracy": ratio(original_correct, total),
            "ephemeral_correct": ephemeral_correct,
            "ephemeral_accuracy": ratio(ephemeral_correct, total),
            "paired_both_correct": transitions["both_correct"],
            "paired_both_correct_rate": ratio(transitions["both_correct"], total),
            "transitions": dict(sorted(transitions.items())),
            "net_accuracy_change": ratio(ephemeral_correct - original_correct, total),
            "discordant_pair_sign_test_p": paired_sign_test_p(
                transitions["original_correct_ephemeral_wrong"],
                transitions["original_wrong_ephemeral_correct"],
            ),
            "degradation_error_counts": dict(sorted(degradation_errors.items())),
            "improved_original_error_counts": dict(sorted(improvement_errors.items())),
            "by_tool_group": {},
        }
        for group, counts in sorted(grouped_transitions.items()):
            group_total = sum(counts.values())
            group_original = counts["both_correct"] + counts["original_correct_ephemeral_wrong"]
            group_ephemeral = counts["both_correct"] + counts["original_wrong_ephemeral_correct"]
            summary["methods"][method]["by_tool_group"][group] = {
                "pairs": group_total,
                "original_accuracy": ratio(group_original, group_total),
                "ephemeral_accuracy": ratio(group_ephemeral, group_total),
                "net_accuracy_change": ratio(group_ephemeral - group_original, group_total),
                "transitions": dict(sorted(counts.items())),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "interface_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "interface_pair_cases.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
