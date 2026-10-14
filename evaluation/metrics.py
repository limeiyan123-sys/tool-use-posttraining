"""严格工具调用解析与可验证指标实现。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Any


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ParseResult:
    """保存严格解析结果。"""

    valid: bool
    calls: list[dict[str, Any]]
    error: str | None


def parse_tool_calls(text: str) -> ParseResult:
    """严格解析所有 tool_call 标签，不尝试修复模型输出。"""
    open_count = text.count("<tool_call>")
    close_count = text.count("</tool_call>")
    if open_count != close_count:
        return ParseResult(False, [], "tool_call 标签未闭合")
    if open_count == 0:
        return ParseResult(True, [], None)

    blocks = TOOL_CALL_PATTERN.findall(text)
    if len(blocks) != open_count:
        return ParseResult(False, [], "tool_call 标签结构非法")

    calls: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        try:
            value = json.loads(block)
        except json.JSONDecodeError as error:
            return ParseResult(False, [], f"第 {index + 1} 个调用不是合法 JSON：{error.msg}")
        if not isinstance(value, dict):
            return ParseResult(False, [], f"第 {index + 1} 个调用不是 JSON 对象")
        if set(value) != {"name", "arguments"}:
            return ParseResult(False, [], f"第 {index + 1} 个调用必须且只能包含 name、arguments")
        if not isinstance(value["name"], str) or not value["name"]:
            return ParseResult(False, [], f"第 {index + 1} 个调用的 name 非法")
        if not isinstance(value["arguments"], dict):
            return ParseResult(False, [], f"第 {index + 1} 个调用的 arguments 不是对象")
        calls.append(value)
    return ParseResult(True, calls, None)


def values_equal(left: Any, right: Any) -> bool:
    """比较参数值；数值允许 int/float 的等值表示，其余类型严格比较。"""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return type(left) is type(right) and left == right


def argument_score(predicted: dict[str, Any], expected: dict[str, list[Any]]) -> tuple[float, bool]:
    """计算单个调用的参数字段准确率与精确匹配结果。"""
    keys = set(predicted) | set(expected)
    if not keys:
        return 1.0, True
    correct = 0
    for key in keys:
        candidates = expected.get(key)
        if candidates is None:
            continue
        if key not in predicted:
            if "" in candidates:
                correct += 1
            continue
        if any(values_equal(predicted[key], candidate) for candidate in candidates):
            correct += 1
    score = correct / len(keys)
    return score, correct == len(keys)


def align_calls(
    predicted: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> tuple[bool, float, bool]:
    """按函数名进行最优对齐，处理并行调用顺序不同的情况。"""
    predicted_names = Counter(call["name"] for call in predicted)
    expected_names = Counter(call["name"] for call in expected)
    tool_exact = predicted_names == expected_names
    if not tool_exact:
        return False, 0.0, False
    if not expected:
        return True, 1.0, True

    scores: list[float] = []
    exact_flags: list[bool] = []
    grouped_predicted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_expected: dict[str, list[dict[str, list[Any]]]] = defaultdict(list)
    for call in predicted:
        grouped_predicted[call["name"]].append(call["arguments"])
    for call in expected:
        grouped_expected[call["name"]].append(call["arguments"])

    for name, expected_group in grouped_expected.items():
        predicted_group = grouped_predicted[name]
        best: tuple[float, list[bool]] | None = None
        for ordering in permutations(predicted_group):
            pair_results = [argument_score(pred, exp) for pred, exp in zip(ordering, expected_group)]
            total = sum(score for score, _ in pair_results)
            flags = [flag for _, flag in pair_results]
            if best is None or total > best[0]:
                best = (total, flags)
        if best is None:
            raise RuntimeError(f"函数 {name} 的调用对齐失败")
        scores.append(best[0] / len(expected_group))
        exact_flags.extend(best[1])

    return True, sum(scores) / len(scores), all(exact_flags)


def validate_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    """验证单个 JSON 值是否满足本项目所需的 JSON Schema 子集。"""
    errors: list[str] = []
    expected_type = schema.get("type")
    valid_type = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "array": lambda item: isinstance(item, list),
        "object": lambda item: isinstance(item, dict),
    }
    if expected_type in valid_type and not valid_type[expected_type](value):
        return [f"{path} 类型应为 {expected_type}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} 不在枚举范围内")
    if expected_type == "array" and isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate_value(item, schema["items"], f"{path}[{index}]"))
    if expected_type == "object" and isinstance(value, dict):
        errors.extend(validate_arguments(value, schema, path))
    return errors


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any], path: str = "arguments") -> list[str]:
    """验证参数对象的必填字段、额外字段与递归类型。"""
    errors: list[str] = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            errors.append(f"{path}.{key} 缺失")
    if schema.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                errors.append(f"{path}.{key} 是未声明参数")
    for key, value in arguments.items():
        if key in properties:
            errors.extend(validate_value(value, properties[key], f"{path}.{key}"))
    return errors


def schema_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """提取工具名到参数 Schema 的映射。"""
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool["function"]
        name = function["name"]
        if name in result:
            raise ValueError(f"工具名重复：{name}")
        result[name] = function["parameters"]
    return result


def evaluate_case(sample: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """评测单条预测并给出细粒度错误类型。"""
    raw_response = prediction.get("response")
    if not isinstance(raw_response, str):
        raise TypeError(f"预测 {sample['id']} 缺少字符串 response")
    parsed = parse_tool_calls(raw_response)
    expected = sample["expected_calls"]
    if not parsed.valid:
        return {
            "id": sample["id"],
            "track": sample["track"],
            "category": sample["category"],
            "parse_valid": False,
            "tool_correct": False,
            "argument_accuracy": 0.0,
            "argument_exact": False,
            "execution_success": False if sample["executable"] else None,
            "error_type": "parse_error",
            "error_detail": parsed.error,
        }

    tool_correct, arg_accuracy, arg_exact = align_calls(parsed.calls, expected)
    schemas = schema_by_name(sample["tools"])
    schema_errors: list[str] = []
    for call in parsed.calls:
        if call["name"] not in schemas:
            schema_errors.append(f"未知工具：{call['name']}")
        else:
            schema_errors.extend(validate_arguments(call["arguments"], schemas[call["name"]]))

    execution_success: bool | None = None
    if sample["executable"]:
        execution_success = tool_correct and arg_exact and not schema_errors

    if not tool_correct:
        error_type = "tool_selection_error"
    elif not arg_exact:
        error_type = "argument_error"
    elif schema_errors:
        error_type = "schema_validation_error"
    else:
        error_type = "none"

    return {
        "id": sample["id"],
        "track": sample["track"],
        "category": sample["category"],
        "parse_valid": True,
        "tool_correct": tool_correct,
        "argument_accuracy": arg_accuracy,
        "argument_exact": arg_exact,
        "execution_success": execution_success,
        "error_type": error_type,
        "error_detail": schema_errors or None,
        "predicted_calls": parsed.calls,
        "expected_calls": expected,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合总体、轨道和类别指标。"""
    if not results:
        raise ValueError("没有可聚合的评测结果")

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        executable = [item for item in group if item["execution_success"] is not None]
        errors = Counter(item["error_type"] for item in group)
        return {
            "samples": len(group),
            "parse_rate": sum(item["parse_valid"] for item in group) / len(group),
            "tool_accuracy": sum(item["tool_correct"] for item in group) / len(group),
            "argument_accuracy": sum(item["argument_accuracy"] for item in group) / len(group),
            "argument_exact_match": sum(item["argument_exact"] for item in group) / len(group),
            "execution_success_rate": (
                sum(item["execution_success"] for item in executable) / len(executable) if executable else None
            ),
            "executable_samples": len(executable),
            "error_counts": dict(sorted(errors.items())),
        }

    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_track[result["track"]].append(result)
        by_category[f"{result['track']}::{result['category']}"] .append(result)
    return {
        "overall": summarize(results),
        "by_track": {key: summarize(value) for key, value in sorted(by_track.items())},
        "by_category": {key: summarize(value) for key, value in sorted(by_category.items())},
    }
