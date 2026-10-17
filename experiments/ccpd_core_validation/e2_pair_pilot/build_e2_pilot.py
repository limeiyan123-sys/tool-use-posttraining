#!/usr/bin/env python3
"""从既有 FCTW 来源构造无“是否变异”捷径的 CCPD E2 最小合同配对。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析数据源、输出位置和 Pilot 规模。"""
    parser = argparse.ArgumentParser(description="构造 CCPD E2 高可信合同配对 Pilot")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261017)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """稳定写出 JSONL，并拒绝覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def stable_name(base_id: str, side: str) -> str:
    """生成等长不透明工具名，避免名称直接泄露标签。"""
    suffix = hashlib.sha256(f"{base_id}:{side}".encode("utf-8")).hexdigest()[:12]
    return f"tool_{suffix}"


def value_in_query(value: Any, query: str) -> bool:
    """保守检查 Gold 标量是否在用户请求中具有直接文本证据。"""
    normalized = query.casefold()
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        text = format(value, "g")
        return re.search(rf"(?<![0-9.]){re.escape(text)}(?![0-9.])", normalized) is not None
    if isinstance(value, str):
        return value.strip().casefold() in normalized
    return False


def has_strict_upper_bound_ambiguity(query: str, values: list[Any], family: str) -> bool:
    """排除自然语言严格小于与 Schema maximum 闭区间冲突的样本。"""
    if family != "numeric_range":
        return False
    normalized = query.casefold()
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        rendered = re.escape(format(value, "g"))
        pattern = rf"(?:under|less\s+than|below)\s+{rendered}(?![0-9.])"
        if re.search(pattern, normalized):
            return True
    return False


def flatten_target_values(expected_calls: list[dict[str, Any]], target: str) -> list[Any]:
    """提取每个 Gold 调用目标字段的首选标量值。"""
    values: list[Any] = []
    for call in expected_calls:
        candidates = call["arguments"].get(target)
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Gold 调用缺少目标字段 {target}")
        value = candidates[0]
        if isinstance(value, (dict, list)) or value is None or isinstance(value, bool):
            raise ValueError(f"目标字段 {target} 不是受支持的标量")
        values.append(value)
    return values


def violates(value: Any, constraint: dict[str, Any]) -> bool:
    """判断标量值是否违反 Pilot 支持的单一约束。"""
    if len(constraint) != 1:
        return False
    key, limit = next(iter(constraint.items()))
    if key == "enum":
        return value not in limit
    if key == "maximum":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > limit
    return False


def positive_constraint(family: str, values: list[Any], negative: dict[str, Any]) -> dict[str, Any]:
    """构造与负侧同类型、但包含全部 Gold 值的中性约束。"""
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if family == "enum":
        return {"enum": unique}
    if family == "numeric_range" and "maximum" in negative:
        maximum = max(float(value) for value in unique)
        if all(isinstance(value, int) and not isinstance(value, bool) for value in unique):
            maximum = int(maximum)
        return {"maximum": maximum}
    raise ValueError(f"不支持的约束族或方向：{family} / {negative}")


def render_constraint(constraint: dict[str, Any]) -> str:
    """用两侧相同模板渲染能力约束。"""
    key, value = next(iter(constraint.items()))
    if key == "enum":
        rendered = ", ".join(json.dumps(item, ensure_ascii=False) for item in value)
        return f"This contract restricts the field to the following values: {rendered}."
    if key == "maximum":
        return f"This contract restricts the field to values at most {value}."
    raise ValueError(f"不支持的约束：{key}")


def apply_contract(
    source_tool: dict[str, Any], base_id: str, side: str, target: str, constraint: dict[str, Any]
) -> dict[str, Any]:
    """从同一合法工具复制并施加一处约束，确保两侧编辑方式一致。"""
    tool = copy.deepcopy(source_tool)
    function = tool["function"]
    function["name"] = stable_name(base_id, side)
    clause = render_constraint(constraint)
    function["description"] = f"{function['description'].rstrip()} {clause}"
    parameters = function["parameters"]
    field = parameters["properties"][target]
    field["description"] = f"{field.get('description', '').rstrip()} {clause}".strip()
    field.update(copy.deepcopy(constraint))
    return tool


def schema_accepts(arguments: dict[str, Any], tool: dict[str, Any]) -> bool:
    """验证 Pilot 实际使用的 required、enum 与 maximum 合同子集。"""
    parameters = tool["function"]["parameters"]
    if any(name not in arguments for name in parameters.get("required", [])):
        return False
    for name, value in arguments.items():
        schema = parameters["properties"].get(name)
        if schema is None:
            return False
        if "enum" in schema and value not in schema["enum"]:
            return False
        if "maximum" in schema:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value > schema["maximum"]:
                return False
    return True


def materialize_call(call: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """把候选值形式的冻结 Gold 转成确定调用，用于合同执行检查。"""
    return {
        "name": tool_name,
        "arguments": {name: candidates[0] for name, candidates in call["arguments"].items()},
    }


def build_candidate(group: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """从一个四象限组生成 E2 配对，失败时返回明确舍弃原因。"""
    if not {"E01", "E10"}.issubset(group):
        return None, "world_incomplete"
    valid_world = group["E10"]
    invalid_world = group["E01"]
    transformation = invalid_world["transformation"]
    target = transformation["target_parameter"]
    family = transformation["constraint_family"]
    negative = transformation["constraint"]
    if family not in {"enum", "numeric_range"}:
        return None, "unsupported_family"
    expected_calls = valid_world["expected_calls"]
    if not expected_calls:
        return None, "missing_gold_call"
    valid_name = valid_world["transformation"]["valid_tool"]
    matches = [tool for tool in valid_world["tools"] if tool["function"]["name"] == valid_name]
    if len(matches) != 1 or any(call["name"] != valid_name for call in expected_calls):
        return None, "multiple_target_tools"
    source_tool = matches[0]
    parameters = source_tool["function"]["parameters"]
    if target not in parameters.get("required", []):
        return None, "target_optional"
    if "default" in parameters["properties"][target]:
        return None, "target_has_default"
    try:
        target_values = flatten_target_values(expected_calls, target)
    except ValueError:
        return None, "unsupported_target_value"
    query = valid_world["messages"][0]["content"]
    if not all(value_in_query(value, query) for value in target_values):
        return None, "no_direct_query_evidence"
    if has_strict_upper_bound_ambiguity(query, target_values, family):
        return None, "strict_upper_bound_ambiguity"
    if not all(violates(value, negative) for value in target_values):
        return None, "negative_does_not_exclude_gold"
    try:
        positive = positive_constraint(family, target_values, negative)
    except ValueError:
        return None, "unsupported_constraint_direction"
    plus_tool = apply_contract(source_tool, valid_world["base_id"], "plus", target, positive)
    minus_tool = apply_contract(source_tool, valid_world["base_id"], "minus", target, negative)
    plus_calls = [materialize_call(call, plus_tool["function"]["name"]) for call in expected_calls]
    minus_calls = [materialize_call(call, minus_tool["function"]["name"]) for call in expected_calls]
    plus_pass = all(schema_accepts(call["arguments"], plus_tool) for call in plus_calls)
    minus_pass = any(schema_accepts(call["arguments"], minus_tool) for call in minus_calls)
    if not plus_pass or minus_pass:
        return None, "contract_execution_check_failed"
    base_id = valid_world["base_id"]
    return {
        "pair_id": f"ccpd_{base_id}",
        "base_id": base_id,
        "source_id": valid_world["source_id"],
        "category": valid_world["category"],
        "constraint_family": family,
        "target_parameter": target,
        "query": query,
        "gold_target_values": target_values,
        "positive_constraint": positive,
        "negative_constraint": negative,
        "plus": {
            "messages": valid_world["messages"],
            "tools": [plus_tool],
            "expected_decision": "call",
            "expected_calls": plus_calls,
        },
        "minus": {
            "messages": valid_world["messages"],
            "tools": [minus_tool],
            "expected_decision": "decline",
            "expected_calls": [],
        },
        "verification": {
            "query_evidence": True,
            "target_required": True,
            "target_has_no_default": True,
            "plus_gold_schema_pass": plus_pass,
            "minus_gold_schema_pass": minus_pass,
            "single_tool_closed_world": True,
            "same_mutation_operator_both_sides": True,
        },
        "limitation": "执行检查覆盖 required/enum/maximum；它验证合同一致性，不是外部业务语义证明。",
    }, "accepted"


def paired_surface_rules(pair: dict[str, Any]) -> dict[str, bool]:
    """检查不读取约束值语义的简单编辑痕迹能否区分两侧。"""
    plus = pair["plus"]["tools"][0]["function"]
    minus = pair["minus"]["tools"][0]["function"]
    plus_json = json.dumps(plus, ensure_ascii=False, sort_keys=True)
    minus_json = json.dumps(minus, ensure_ascii=False, sort_keys=True)
    return {
        "constraint_presence_distinguishes": (
            any(key in plus_json for key in ('"enum"', '"maximum"'))
            != any(key in minus_json for key in ('"enum"', '"maximum"'))
        ),
        "restriction_marker_distinguishes": (
            ("This contract restricts" in plus_json)
            != ("This contract restricts" in minus_json)
        ),
        "tool_name_prefix_distinguishes": (
            plus["name"].split("_")[0] != minus["name"].split("_")[0]
        ),
        "minus_serialized_longer": len(minus_json) > len(plus_json),
    }


def main() -> None:
    """构造、筛选并冻结 60 对 E2 Pilot。"""
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    source_path = root / "data/sft/fctw_v1/worlds.jsonl"
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(source_path):
        groups[row["base_id"]][row["world_type"]] = row
    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for base_id in sorted(groups):
        pair, reason = build_candidate(groups[base_id])
        reasons[reason] += 1
        if pair is not None:
            accepted.append(pair)
    if len(accepted) < args.pairs:
        raise RuntimeError(f"仅生成 {len(accepted)} 对，少于请求的 {args.pairs} 对")
    # 使用哈希排序，避免按原始数据类别或 ID 顺序截断。
    accepted.sort(
        key=lambda row: hashlib.sha256(f"{args.seed}:{row['pair_id']}".encode("utf-8")).hexdigest()
    )
    pilot = accepted[: args.pairs]
    for index, pair in enumerate(pilot):
        pair["split"] = "train" if index < 40 else "dev" if index < 50 else "test"
        pair["surface_shortcut_checks"] = paired_surface_rules(pair)
    shortcut_counts = Counter()
    for pair in pilot:
        for key, value in pair["surface_shortcut_checks"].items():
            shortcut_counts[key] += int(value)
    output.mkdir(parents=True, exist_ok=True)
    pair_path = output / "e2_pilot_pairs.jsonl"
    manifest_path = output / "manifest.json"
    audit_path = output / "generation_audit.json"
    for path in (pair_path, manifest_path, audit_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有 E2 结果：{path}")
    write_jsonl(pair_path, pilot)
    audit = {
        "source_groups": len(groups),
        "candidate_outcomes": dict(sorted(reasons.items())),
        "accepted_before_sampling": len(accepted),
        "pilot_pairs": len(pilot),
        "shortcut_counts": dict(shortcut_counts),
        "manual_review_status": "pending",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": "ccpd_e2_pilot_v1",
        "seed": args.seed,
        "source": str(source_path.relative_to(root)),
        "source_sha256": sha256_file(source_path),
        "pairs": len(pilot),
        "splits": dict(Counter(pair["split"] for pair in pilot)),
        "constraint_families": dict(Counter(pair["constraint_family"] for pair in pilot)),
        "categories": dict(Counter(pair["category"] for pair in pilot)),
        "data_sha256": sha256_file(pair_path),
        "guarantees": [
            "目标字段为必填且没有 default",
            "Gold 目标值在用户请求中具有直接文本证据",
            "E+ 与 E- 从同一工具复制并使用相同类型的约束编辑",
            "E+ Gold 通过合同子集检查，E- Gold 被合同子集检查拒绝",
            "两侧均为单工具闭世界且使用等长不透明名称",
        ],
        "limitations": [
            "来源任务参与过 FCTW 训练数据构造，不能作为历史 Adapter 的独立泛化集",
            "执行检查仅覆盖 required、enum、maximum，不证明真实外部 API 业务语义",
            "当前仅完成机械审计，人工语义复核状态单独记录",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
