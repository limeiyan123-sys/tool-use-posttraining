"""反事实工具世界的严格评测与成对指标。"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from evaluation.metrics import align_calls, evaluate_case, parse_tool_calls


def canonicalize_calls(
    calls: list[dict[str, Any]], semantic_map: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """依据显式接口映射恢复规范语义调用，不修复未知字段。"""
    canonical: list[dict[str, Any]] = []
    errors: list[str] = []
    for call in calls:
        mapping = semantic_map.get(call["name"])
        if mapping is None:
            errors.append(f"未知接口工具：{call['name']}")
            continue
        argument_map = mapping["arguments"]
        unknown = set(call["arguments"]) - set(argument_map)
        if unknown:
            errors.append(f"{call['name']} 包含未知接口参数：{sorted(unknown)}")
            continue
        canonical.append(
            {
                "name": mapping["canonical_name"],
                "arguments": {
                    argument_map[key]: value for key, value in call["arguments"].items()
                },
            }
        )
    return canonical, errors


def classify_tool_world_error(
    sample: dict[str, Any], base_result: dict[str, Any], parsed_calls: list[dict[str, Any]]
) -> str:
    """将通用错误细化为工具世界诊断错误。"""
    if not base_result["parse_valid"]:
        return "parse_error"
    expected = sample["expected_calls"]
    if expected and not parsed_calls:
        return "unnecessary_refusal"
    predicted_names = {call["name"] for call in parsed_calls}
    world_type = sample["world_type"]
    transformation = sample["transformation"]
    available_names = {tool["function"]["name"] for tool in sample["tools"]}
    if world_type == "capability_removal" and predicted_names - available_names:
        return "unavailable_tool_hallucination"
    if (
        world_type == "capability_substitution"
        and transformation.get("replacement_tool") in predicted_names
    ):
        return "incompatible_substitute_selection"
    if world_type == "distractor" and transformation.get("distractor_tool") in predicted_names:
        return "distractor_selection_error"
    if world_type == "ephemeral_interface" and not base_result["tool_correct"]:
        return "interface_tool_mapping_error"
    if not base_result["tool_correct"]:
        return "tool_selection_error"
    if world_type == "ephemeral_interface" and not base_result["argument_exact"]:
        return "interface_argument_mapping_error"
    if not base_result["argument_exact"]:
        return "argument_error"
    if base_result["error_type"] == "schema_validation_error":
        return "schema_validation_error"
    return "none"


def evaluate_tool_world_case(
    sample: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, Any]:
    """评测单个工具世界并附加规范语义等变性结果。"""
    base_result = evaluate_case(sample, prediction)
    parsed = parse_tool_calls(prediction["response"])
    semantic_exact = False
    semantic_errors: list[str] = []
    if parsed.valid:
        canonical_calls, semantic_errors = canonicalize_calls(parsed.calls, sample["semantic_map"])
        if not semantic_errors:
            _, _, semantic_exact = align_calls(
                canonical_calls, sample["canonical_expected_calls"]
            )
    decision_exact = (
        base_result["parse_valid"]
        and base_result["tool_correct"]
        and base_result["argument_exact"]
        and base_result["error_type"] == "none"
    )
    parsed_calls = parsed.calls if parsed.valid else []
    base_result.update(
        {
            "base_id": sample["base_id"],
            "source_id": sample["source_id"],
            "world_type": sample["world_type"],
            "decision_exact": decision_exact,
            "semantic_exact": semantic_exact,
            "tool_world_error": classify_tool_world_error(sample, base_result, parsed_calls),
            "semantic_error_detail": semantic_errors or None,
        }
    )
    return base_result


def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一组工具世界结果。"""
    if not group:
        raise ValueError("没有可汇总的工具世界结果")
    return {
        "samples": len(group),
        "parse_rate": sum(item["parse_valid"] for item in group) / len(group),
        "tool_accuracy": sum(item["tool_correct"] for item in group) / len(group),
        "argument_accuracy": sum(item["argument_accuracy"] for item in group) / len(group),
        "argument_exact_match": sum(item["argument_exact"] for item in group) / len(group),
        "decision_exact": sum(item["decision_exact"] for item in group) / len(group),
        "semantic_exact": sum(item["semantic_exact"] for item in group) / len(group),
        "error_counts": dict(sorted(Counter(item["tool_world_error"] for item in group).items())),
    }


def aggregate_tool_world(results: list[dict[str, Any]]) -> dict[str, Any]:
    """计算分世界指标、接口等变准确率和反事实成对准确率。"""
    if not results:
        raise ValueError("没有可聚合的工具世界结果")
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        by_world[result["world_type"]].append(result)
        if result["world_type"] in by_base[result["base_id"]]:
            raise RuntimeError(f"基础任务 {result['base_id']} 的世界类型重复")
        by_base[result["base_id"]][result["world_type"]] = result

    required_worlds = {
        "original",
        "ephemeral_interface",
        "capability_removal",
        "capability_substitution",
        "distractor",
    }
    for base_id, worlds in by_base.items():
        if set(worlds) != required_worlds:
            raise RuntimeError(f"基础任务 {base_id} 的世界不完整：{sorted(worlds)}")

    pair_accuracy: dict[str, dict[str, Any]] = {}
    for world_type in sorted(required_worlds - {"original"}):
        pairs = [
            worlds["original"]["decision_exact"] and worlds[world_type]["decision_exact"]
            for worlds in by_base.values()
        ]
        pair_accuracy[world_type] = {
            "pairs": len(pairs),
            "both_correct": sum(pairs),
            "accuracy": sum(pairs) / len(pairs),
        }
    all_worlds = [
        all(worlds[world_type]["decision_exact"] for world_type in required_worlds)
        for worlds in by_base.values()
    ]
    ephemeral = by_world["ephemeral_interface"]
    return {
        "overall": summarize(results),
        "by_world": {key: summarize(value) for key, value in sorted(by_world.items())},
        "interface_equivariance_accuracy": (
            sum(item["semantic_exact"] for item in ephemeral) / len(ephemeral)
        ),
        "counterfactual_pair_accuracy": pair_accuracy,
        "all_worlds_group_accuracy": {
            "groups": len(all_worlds),
            "all_correct": sum(all_worlds),
            "accuracy": sum(all_worlds) / len(all_worlds),
        },
    }
