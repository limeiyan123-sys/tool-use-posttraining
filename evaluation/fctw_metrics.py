"""FCTW 四象限的严格指标与错误归因。"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from evaluation.metrics import evaluate_case, parse_tool_calls


def classify_fctw_error(
    sample: dict[str, Any], result: dict[str, Any], response: str
) -> str:
    """将通用评测结果映射为 FCTW 的四类错误。"""
    parsed = parse_tool_calls(response)
    if not parsed.valid:
        return "invalid_output"
    if sample["expected_calls"] and not parsed.calls:
        return "false_rejection"
    if not result["tool_correct"]:
        return "wrong_tool_selection"
    if not result["argument_exact"] or result["error_type"] == "schema_validation_error":
        return "argument_failure"
    return "correct"


def evaluate_fctw_case(
    sample: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, Any]:
    """评测单个 FCTW 世界并附加因子标签。"""
    evaluation_sample = sample if "executable" in sample else {**sample, "executable": False}
    result = evaluate_case(evaluation_sample, prediction)
    decision_exact = (
        result["parse_valid"]
        and result["tool_correct"]
        and result["argument_exact"]
        and result["error_type"] == "none"
    )
    result.update(
        {
            "base_id": sample["base_id"],
            "source_id": sample["source_id"],
            "world_type": sample["world_type"],
            "valid_capability": sample["valid_capability"],
            "near_miss_capability": sample["near_miss_capability"],
            "constraint_family": sample["transformation"]["constraint_family"],
            "decision_exact": decision_exact,
            "fctw_error": classify_fctw_error(sample, result, prediction["response"]),
        }
    )
    return result


def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总准确率与错误率。"""
    if not group:
        raise ValueError("没有可汇总的 FCTW 结果")
    errors = Counter(item["fctw_error"] for item in group)
    return {
        "samples": len(group),
        "parse_rate": sum(item["parse_valid"] for item in group) / len(group),
        "tool_accuracy": sum(item["tool_correct"] for item in group) / len(group),
        "argument_accuracy": sum(item["argument_accuracy"] for item in group) / len(group),
        "argument_exact_match": sum(item["argument_exact"] for item in group) / len(group),
        "decision_exact": sum(item["decision_exact"] for item in group) / len(group),
        "false_rejection_rate": errors["false_rejection"] / len(group),
        "wrong_tool_selection_rate": errors["wrong_tool_selection"] / len(group),
        "argument_failure_rate": errors["argument_failure"] / len(group),
        "invalid_output_rate": errors["invalid_output"] / len(group),
        "error_counts": dict(sorted(errors.items())),
    }


def aggregate_fctw(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按四象限、约束族和意图组聚合 FCTW 指标。"""
    if not results:
        raise ValueError("没有可聚合的 FCTW 结果")
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_constraint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_constraint_world: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        by_world[result["world_type"]].append(result)
        by_constraint[result["constraint_family"]].append(result)
        by_constraint_world[result["constraint_family"]][result["world_type"]].append(result)
        by_base[result["base_id"]][result["world_type"]] = result
    required = {"E10", "E01", "E11", "E00"}
    for base_id, worlds in by_base.items():
        if set(worlds) != required:
            raise RuntimeError(f"{base_id} 的 FCTW 世界不完整：{sorted(worlds)}")
    all_correct = [
        all(worlds[quadrant]["decision_exact"] for quadrant in required)
        for worlds in by_base.values()
    ]
    return {
        "overall": summarize(results),
        "by_world": {key: summarize(value) for key, value in sorted(by_world.items())},
        "by_constraint": {
            key: summarize(value) for key, value in sorted(by_constraint.items())
        },
        "by_constraint_world": {
            family: {
                world: summarize(group) for world, group in sorted(worlds.items())
            }
            for family, worlds in sorted(by_constraint_world.items())
        },
        "all_worlds_group_accuracy": {
            "groups": len(all_correct),
            "all_correct": sum(all_correct),
            "accuracy": sum(all_correct) / len(all_correct),
        },
        "capability_substitution_accuracy": summarize(by_world["E01"])["decision_exact"],
    }
