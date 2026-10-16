#!/usr/bin/env python3
"""对 Stage 6 原始预测与 Schema 过滤结果进行可追溯的 Stage 8 错误归因。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any


METHOD_PATHS = {
    "FDSRC": {
        "raw": "reports/tool_world_final/fdsrc_predictions.jsonl",
        "filtered": (
            "reports/stage6_schema_grounded/symbolic_filter/fdsrc/"
            "fdsrc_filtered_predictions.jsonl"
        ),
    },
    "FWCT": {
        "raw": "reports/tool_world_final/failure_weighted_predictions.jsonl",
        "filtered": (
            "reports/stage6_schema_grounded/symbolic_filter/fwct/"
            "fwct_filtered_predictions.jsonl"
        ),
    },
    "FCTW": {
        "raw": "reports/stage5_fctw/tool_world_final/fctw_predictions.jsonl",
        "filtered": (
            "reports/stage6_schema_grounded/symbolic_filter/fctw/"
            "fctw_filtered_predictions.jsonl"
        ),
    },
}

DATASET_PATH = "data/evaluation/tool_world_final_v1/tool_world_final_v1.jsonl"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Stage 8 剩余错误归因")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL，不接受空行或非对象记录。"""
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
    """以稳定字段顺序写入 JSONL。"""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prediction_map(rows: list[dict[str, Any]], expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    """校验预测 ID 集合并建立索引。"""
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("预测文件存在重复 ID")
    if set(result) != expected_ids:
        missing = sorted(expected_ids - set(result))
        extra = sorted(set(result) - expected_ids)
        raise RuntimeError(f"预测 ID 不匹配，缺失={missing[:3]}，额外={extra[:3]}")
    return result


def schema_errors(sample: dict[str, Any], calls: list[dict[str, Any]], validate_arguments: Any) -> list[str]:
    """返回调用相对当前工具环境的确定性 Schema 错误。"""
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in sample["tools"]
    }
    errors: list[str] = []
    for call in calls:
        schema = schemas.get(call["name"])
        if schema is None:
            errors.append(f"未知工具：{call['name']}")
            continue
        errors.extend(validate_arguments(call["arguments"], schema))
    return errors


def expected_by_name(sample: dict[str, Any]) -> dict[str, list[dict[str, list[Any]]]]:
    """按工具名整理 Gold 参数候选。"""
    grouped: dict[str, list[dict[str, list[Any]]]] = defaultdict(list)
    for call in sample["expected_calls"]:
        grouped[call["name"]].append(call["arguments"])
    return grouped


def argument_mismatch_detail(
    sample: dict[str, Any], calls: list[dict[str, Any]], argument_score: Any, values_equal: Any
) -> dict[str, Any]:
    """在同名重复调用的最优对齐下拆分缺失字段、错误值与额外字段。"""
    expected = expected_by_name(sample)
    predicted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        predicted[call["name"]].append(call["arguments"])
    details: list[dict[str, Any]] = []
    for name, expected_group in expected.items():
        predicted_group = predicted.get(name, [])
        if len(predicted_group) != len(expected_group):
            continue
        best_ordering: tuple[dict[str, Any], ...] | None = None
        best_score = -1.0
        for ordering in permutations(predicted_group):
            score = sum(
                argument_score(predicted_args, expected_args)[0]
                for predicted_args, expected_args in zip(ordering, expected_group)
            )
            if score > best_score:
                best_score = score
                best_ordering = ordering
        if best_ordering is None:
            continue
        for expected_args, predicted_args in zip(expected_group, best_ordering):
            mandatory = {
                key for key, candidates in expected_args.items() if "" not in candidates
            }
            missing = sorted(mandatory - set(predicted_args))
            unexpected = sorted(set(predicted_args) - set(expected_args))
            wrong_values = sorted(
                key
                for key in set(predicted_args) & set(expected_args)
                if not any(values_equal(predicted_args[key], value) for value in expected_args[key])
            )
            if missing or unexpected or wrong_values:
                details.append(
                    {
                        "tool": name,
                        "missing_required_gold_keys": missing,
                        "unexpected_keys": unexpected,
                        "wrong_value_keys": wrong_values,
                    }
                )
    return {
        "calls_with_mismatch": details,
        "has_requirement_omission": any(
            detail["missing_required_gold_keys"] for detail in details
        ),
        "has_wrong_value": any(detail["wrong_value_keys"] for detail in details),
        "has_unexpected_key": any(detail["unexpected_keys"] for detail in details),
    }


def classify_observable(
    sample: dict[str, Any],
    prediction: dict[str, Any],
    evaluate_tool_world_case: Any,
    parse_tool_calls: Any,
    validate_arguments: Any,
    argument_score: Any,
    values_equal: Any,
) -> dict[str, Any]:
    """生成主要错误、次要错误与确定性证据，不推断内部机制。"""
    result = evaluate_tool_world_case(sample, prediction)
    parsed = parse_tool_calls(prediction["response"])
    if not parsed.valid:
        return {
            "decision_exact": False,
            "primary_failure": "output_or_parsing_error",
            "secondary_failures": [],
            "parse_valid": False,
            "parsed_calls": [],
            "schema_errors": [],
            "argument_mismatch": None,
            "legal_but_semantically_wrong": False,
            "evaluator_error_type": result["tool_world_error"],
        }

    calls = parsed.calls
    expected_calls = sample["expected_calls"]
    violations = schema_errors(sample, calls, validate_arguments)
    mismatch = argument_mismatch_detail(sample, calls, argument_score, values_equal)
    expected_name_counts = Counter(call["name"] for call in expected_calls)
    predicted_name_counts = Counter(call["name"] for call in calls)
    cardinality_detail = {
        "expected": dict(sorted(expected_name_counts.items())),
        "predicted": dict(sorted(predicted_name_counts.items())),
    }
    secondary: list[str] = []
    primary = "correct"
    if expected_calls and not calls:
        primary = "false_reject"
    elif not expected_calls and calls:
        primary = "false_call"
    elif expected_calls and calls and not result["tool_correct"]:
        if set(predicted_name_counts) <= set(expected_name_counts):
            primary = "call_cardinality_error"
        else:
            primary = "tool_selection_error"
    elif expected_calls and calls and not result["argument_exact"]:
        if mismatch["has_requirement_omission"]:
            primary = "requirement_omission"
        else:
            primary = "argument_grounding_error"
    elif violations:
        primary = "schema_constraint_error"

    if violations and primary != "schema_constraint_error":
        secondary.append("schema_constraint_error")
    if expected_calls and calls and mismatch["has_requirement_omission"] and primary != "requirement_omission":
        secondary.append("requirement_omission")
    legal_but_wrong = bool(
        parsed.valid and calls and not violations and not result["decision_exact"]
    )
    return {
        "decision_exact": bool(result["decision_exact"]),
        "primary_failure": primary,
        "secondary_failures": sorted(set(secondary)),
        "parse_valid": True,
        "parsed_calls": calls,
        "schema_errors": violations,
        "argument_mismatch": mismatch,
        "call_cardinality": cardinality_detail,
        "legal_but_semantically_wrong": legal_but_wrong,
        "evaluator_error_type": result["tool_world_error"],
    }


def find_tool(sample: dict[str, Any], name: str | None) -> dict[str, Any] | None:
    """按函数名返回工具定义。"""
    for tool in sample["tools"]:
        if tool["function"]["name"] == name:
            return tool
    return None


def audit_gold(
    sample: dict[str, Any], original_by_base: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """对高风险构造规则进行保守 Gold 审计，不擅自改写标注。"""
    flags: list[str] = []
    notes: list[str] = []
    confidence = "high"
    transformation = sample.get("transformation", {})
    original = original_by_base[sample["base_id"]]

    if sample["world_type"] == "capability_substitution":
        restricted = transformation.get("restricted_parameter")
        replacement = transformation.get("replacement_tool")
        tool = find_tool(sample, replacement)
        if tool is None:
            flags.append("replacement_tool_missing")
            confidence = "low"
        elif isinstance(restricted, str):
            parameters = tool["function"]["parameters"]
            property_schema = parameters.get("properties", {}).get(restricted, {})
            if restricted not in parameters.get("required", []):
                flags.append("optional_restricted_parameter")
                notes.append("受限字段为可选参数；省略后的真实工具行为未由 Mock 执行结果证明。")
                confidence = "medium"
            description = str(property_schema.get("description", "")).lower()
            if "default" in property_schema or "default" in description:
                flags.append("default_behavior_unclear")
                notes.append("Schema/default 文本不能证明运行时会如何填充或筛选。")
                confidence = "medium"

    original_calls = original["expected_calls"]
    current_calls = sample["expected_calls"]
    available_names = {tool["function"]["name"] for tool in sample["tools"]}
    original_names = {call["name"] for call in original_calls}
    still_available = original_names & available_names
    if len(original_calls) > 1 and not current_calls and still_available:
        flags.append("partial_completion_protocol_unspecified")
        notes.append("多请求任务仍有部分原始工具可用，但 Gold 要求整体拒绝；协议未明确是否允许部分完成。")
        confidence = "low"

    return {
        "id": sample["id"],
        "base_id": sample["base_id"],
        "world_type": sample["world_type"],
        "confidence": confidence,
        "flags": sorted(set(flags)),
        "notes": notes,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    """计算带显式分母的比例。"""
    return numerator / denominator if denominator else None


def summarize(records: list[dict[str, Any]], state: str) -> dict[str, Any]:
    """汇总指定状态下的错误类型、场景和语义合法性。"""
    state_rows = [row[state] for row in records]
    errors = Counter(row["primary_failure"] for row in state_rows)
    wrong = [row for row in state_rows if not row["decision_exact"]]
    legal_wrong = sum(row["legal_but_semantically_wrong"] for row in wrong)
    by_world: dict[str, Any] = {}
    for world in sorted({row["world_type"] for row in records}):
        selected_records = [row for row in records if row["world_type"] == world]
        selected = [row[state] for row in selected_records]
        world_wrong = [row for row in selected if not row["decision_exact"]]
        by_world[world] = {
            "samples": len(selected),
            "correct": sum(row["decision_exact"] for row in selected),
            "accuracy": ratio(sum(row["decision_exact"] for row in selected), len(selected)),
            "errors": len(world_wrong),
            "error_counts": dict(sorted(Counter(row["primary_failure"] for row in world_wrong).items())),
        }
    return {
        "samples": len(state_rows),
        "correct": sum(row["decision_exact"] for row in state_rows),
        "accuracy": ratio(sum(row["decision_exact"] for row in state_rows), len(state_rows)),
        "errors": len(wrong),
        "error_counts": dict(sorted(errors.items())),
        "legal_but_semantically_wrong": legal_wrong,
        "legal_but_semantically_wrong_rate_among_errors": ratio(legal_wrong, len(wrong)),
        "by_world": by_world,
    }


def main() -> None:
    """执行 Stage 8 离线归因并写出逐样本与统计产物。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))
    from evaluation.metrics import argument_score, parse_tool_calls, validate_arguments, values_equal
    from evaluation.tool_world_metrics import evaluate_tool_world_case

    samples = read_jsonl(project_root / DATASET_PATH)
    sample_by_id = {row["id"]: row for row in samples}
    expected_ids = set(sample_by_id)
    if len(expected_ids) != len(samples):
        raise RuntimeError("评测集存在重复 ID")
    original_by_base = {
        row["base_id"]: row for row in samples if row["world_type"] == "original"
    }
    gold_audit = {row["id"]: audit_gold(row, original_by_base) for row in samples}

    all_records: dict[str, list[dict[str, Any]]] = {}
    detailed_errors: list[dict[str, Any]] = []
    for method, paths in METHOD_PATHS.items():
        raw = prediction_map(read_jsonl(project_root / paths["raw"]), expected_ids)
        filtered = prediction_map(read_jsonl(project_root / paths["filtered"]), expected_ids)
        method_records: list[dict[str, Any]] = []
        for sample in samples:
            raw_result = classify_observable(
                sample,
                raw[sample["id"]],
                evaluate_tool_world_case,
                parse_tool_calls,
                validate_arguments,
                argument_score,
                values_equal,
            )
            filtered_result = classify_observable(
                sample,
                filtered[sample["id"]],
                evaluate_tool_world_case,
                parse_tool_calls,
                validate_arguments,
                argument_score,
                values_equal,
            )
            record = {
                "id": sample["id"],
                "base_id": sample["base_id"],
                "method": method,
                "world_type": sample["world_type"],
                "gold_confidence": gold_audit[sample["id"]]["confidence"],
                "raw": raw_result,
                "filtered": filtered_result,
                "blocked_calls": filtered[sample["id"]].get("symbolic_filter", {}).get("blocked_calls", []),
            }
            method_records.append(record)
            if not raw_result["decision_exact"] or not filtered_result["decision_exact"] or record["blocked_calls"]:
                detailed_errors.append(
                    {
                        **record,
                        "query": sample["messages"],
                        "tools": sample["tools"],
                        "gold": sample["expected_calls"],
                        "raw_response": raw[sample["id"]]["response"],
                        "filtered_response": filtered[sample["id"]]["response"],
                        "gold_audit": gold_audit[sample["id"]],
                    }
                )
        all_records[method] = method_records

    statistics: dict[str, Any] = {"methods": {}}
    for method, records in all_records.items():
        wrong_to_correct = sum(
            not row["raw"]["decision_exact"] and row["filtered"]["decision_exact"]
            for row in records
        )
        correct_to_wrong = sum(
            row["raw"]["decision_exact"] and not row["filtered"]["decision_exact"]
            for row in records
        )
        statistics["methods"][method] = {
            "raw": summarize(records, "raw"),
            "filtered": summarize(records, "filtered"),
            "filtered_high_confidence_gold": summarize(
                [row for row in records if row["gold_confidence"] == "high"], "filtered"
            ),
            "affected_samples": sum(bool(row["blocked_calls"]) for row in records),
            "blocked_calls": sum(len(row["blocked_calls"]) for row in records),
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
        }

    wrong_sets = {
        method: {row["id"] for row in records if not row["filtered"]["decision_exact"]}
        for method, records in all_records.items()
    }
    common_wrong = set.intersection(*wrong_sets.values())
    union_wrong = set.union(*wrong_sets.values())
    statistics["cross_method"] = {
        "all_three_common_filtered_errors": len(common_wrong),
        "all_three_common_rate_over_dataset": ratio(len(common_wrong), len(samples)),
        "all_three_common_rate_over_error_union": ratio(len(common_wrong), len(union_wrong)),
        "filtered_error_union": len(union_wrong),
        "pairwise_common": {
            f"{left}_{right}": len(wrong_sets[left] & wrong_sets[right])
            for left, right in (("FDSRC", "FWCT"), ("FDSRC", "FCTW"), ("FWCT", "FCTW"))
        },
    }
    gold_counts = Counter(item["confidence"] for item in gold_audit.values())
    flag_counts = Counter(flag for item in gold_audit.values() for flag in item["flags"])
    statistics["gold_audit"] = {
        "samples": len(samples),
        "confidence_counts": dict(sorted(gold_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
    }

    diagnostic_candidates: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["id"]
        filtered_by_method = {
            method: next(row for row in records if row["id"] == sample_id)["filtered"]
            for method, records in all_records.items()
        }
        if sample_id not in common_wrong:
            continue
        if gold_audit[sample_id]["confidence"] != "high":
            continue
        if not all(item["legal_but_semantically_wrong"] for item in filtered_by_method.values()):
            continue
        diagnostic_candidates.append(
            {
                "id": sample_id,
                "base_id": sample["base_id"],
                "world_type": sample["world_type"],
                "query": sample["messages"],
                "tools": sample["tools"],
                "gold": sample["expected_calls"],
                "filtered_failures": filtered_by_method,
            }
        )

    write_jsonl(output_dir / "remaining_errors.jsonl", detailed_errors)
    write_jsonl(
        output_dir / "all_case_attribution.jsonl",
        [row for method in METHOD_PATHS for row in all_records[method]],
    )
    write_jsonl(
        output_dir / "gold_review.jsonl",
        [gold_audit[row["id"]] for row in samples if gold_audit[row["id"]]["flags"]],
    )
    write_jsonl(output_dir / "diagnostic_candidates.jsonl", diagnostic_candidates)
    (output_dir / "error_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"高置信度共同合法语义错误候选：{len(diagnostic_candidates)}")


if __name__ == "__main__":
    main()
