#!/usr/bin/env python3
"""审计 E1 合同配对，并生成严格主分析集与可选字段敏感性集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="审计 CCPD E1 合同配对")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL，不跳过空行或非法对象。"""
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
    """以稳定顺序写出 JSONL，并拒绝意外覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """计算文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_world_tool(row: dict[str, Any], key: str) -> dict[str, Any]:
    """根据 transformation 中的名称取得唯一目标工具。"""
    name = row["transformation"][key]
    matches = [tool for tool in row["tools"] if tool["function"]["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"{row['id']} 的 {key} 工具数量为 {len(matches)}")
    return matches[0]


def check_constraint_excludes(value: Any, constraint: dict[str, Any]) -> bool:
    """验证单个确定性约束是否排除冻结 Gold 值。"""
    if len(constraint) != 1:
        return False
    key, expected = next(iter(constraint.items()))
    if key == "enum":
        return value not in expected
    if key == "maximum":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > expected
    if key == "minimum":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value < expected
    if key == "pattern":
        import re

        return isinstance(value, str) and re.fullmatch(expected, value) is None
    raise ValueError(f"不支持的约束类型：{key}")


def build_record(
    base_id: str,
    valid_probe: dict[str, Any],
    invalid_probe: dict[str, Any],
    valid_world: dict[str, Any],
    invalid_world: dict[str, Any],
) -> dict[str, Any]:
    """校验一个合同对并生成完整审计记录。"""
    valid_query = valid_probe["messages"][0]["content"]
    invalid_query = invalid_probe["messages"][0]["content"]
    if valid_query != invalid_query:
        raise ValueError(f"{base_id} 两侧请求不一致")
    if len(valid_probe["tools"]) != 1 or len(invalid_probe["tools"]) != 1:
        raise ValueError(f"{base_id} 不是单工具配对")

    target = invalid_world["transformation"]["target_parameter"]
    constraint = invalid_world["transformation"]["constraint"]
    valid_tool = find_world_tool(valid_world, "valid_tool")
    invalid_tool = find_world_tool(invalid_world, "near_miss_tool")
    valid_parameters = valid_tool["function"]["parameters"]
    invalid_parameters = invalid_tool["function"]["parameters"]
    valid_property = valid_parameters["properties"][target]
    invalid_property = invalid_parameters["properties"][target]
    expected_calls = valid_world["expected_calls"]
    if not expected_calls or any(call["name"] != valid_tool["function"]["name"] for call in expected_calls):
        raise ValueError(f"{base_id} 的 Gold 调用缺失或指向非目标工具")
    candidates: list[Any] = []
    for call in expected_calls:
        values = call["arguments"].get(target)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{base_id} 的目标字段缺少 Gold 候选值")
        candidates.extend(values)
    excluded = all(check_constraint_excludes(value, constraint) for value in candidates)
    target_required = target in valid_parameters.get("required", [])
    optional_default = valid_property.get("default")
    return {
        "base_id": base_id,
        "source_id": valid_probe["source_id"],
        "category": valid_probe["category"],
        "constraint_family": invalid_probe["constraint_family"],
        "query": valid_query,
        "target_parameter": target,
        "target_required": target_required,
        "optional_default": optional_default,
        "gold_target_values": candidates,
        "constraint": constraint,
        "constraint_excludes_all_gold_values": excluded,
        "valid_sample": valid_probe,
        "invalid_sample": invalid_probe,
        "audit_tier": "strict_required" if target_required and excluded else "optional_or_rejected",
        "audit_notes": (
            []
            if target_required and excluded
            else [
                *([] if excluded else ["约束没有排除全部冻结 Gold 值"]),
                *([] if target_required else ["目标字段可选，可能受默认行为或省略参数影响"]),
            ]
        ),
    }


def main() -> None:
    """执行全量机械审计并冻结 E1 两个分析层级。"""
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    probe_path = root / "data/evaluation/capability_probe_v1/capability_probe_v1.jsonl"
    world_path = root / "data/evaluation/fctw_eval_v1/fctw_eval_v1.jsonl"
    probe_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    world_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(probe_path):
        if row["sample_type"] in {"valid", "near_miss"}:
            probe_groups[row["base_id"]][row["sample_type"]] = row
    for row in read_jsonl(world_path):
        world_groups[row["base_id"]][row["world_type"]] = row

    records: list[dict[str, Any]] = []
    for base_id in sorted(probe_groups):
        probes = probe_groups[base_id]
        worlds = world_groups[base_id]
        if set(probes) != {"valid", "near_miss"} or not {"E01", "E10"}.issubset(worlds):
            raise RuntimeError(f"{base_id} 的配对数据不完整")
        records.append(
            build_record(base_id, probes["valid"], probes["near_miss"], worlds["E10"], worlds["E01"])
        )

    strict = [row for row in records if row["audit_tier"] == "strict_required"]
    optional = [row for row in records if row["audit_tier"] != "strict_required"]
    if len(records) != 30 or len(strict) != 21 or len(optional) != 9:
        raise RuntimeError(
            f"审计数量偏离预期：all={len(records)}, strict={len(strict)}, optional={len(optional)}"
        )
    for path in (
        output / "pair_audit.jsonl",
        output / "strict_pairs.jsonl",
        output / "optional_pairs.jsonl",
        output / "audit_summary.json",
    ):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有审计结果：{path}")
    write_jsonl(output / "pair_audit.jsonl", records)
    write_jsonl(output / "strict_pairs.jsonl", strict)
    write_jsonl(output / "optional_pairs.jsonl", optional)
    summary = {
        "source_probe": str(probe_path.relative_to(root)),
        "source_probe_sha256": sha256_file(probe_path),
        "source_world": str(world_path.relative_to(root)),
        "source_world_sha256": sha256_file(world_path),
        "pairs_total": len(records),
        "strict_required_pairs": len(strict),
        "optional_sensitivity_pairs": len(optional),
        "strict_constraint_families": dict(Counter(row["constraint_family"] for row in strict)),
        "optional_base_ids": [row["base_id"] for row in optional],
        "exclusion_rule": "主分析仅保留目标字段必填且约束排除全部冻结 Gold 值的配对",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
