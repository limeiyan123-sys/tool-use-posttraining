#!/usr/bin/env python3
"""构建 Valid × Near-miss 四象限的 FCTW-SFT 训练数据。"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_counterfactual_sft import (
    CATEGORY_POOL_SIZES,
    materialize_arguments,
    stable_shuffle,
    world_to_sharegpt,
    write_dataset,
)
from build_frozen_eval import read_jsonl, sha256_file, write_jsonl
from build_tool_world_eval import build_base_tasks, stable_alias, stable_digest


QUADRANTS = {
    "E10": {"valid": True, "near_miss": False, "decision": "call"},
    "E01": {"valid": False, "near_miss": True, "decision": "reject"},
    "E11": {"valid": True, "near_miss": True, "decision": "call"},
    "E00": {"valid": False, "near_miss": False, "decision": "reject"},
}
CONSTRAINT_QUOTAS = {"enum": 50, "numeric_range": 50}
SCALAR_TYPES = {"string", "integer", "number", "boolean"}
REMOVED_CONSTRAINT_KEYS = {
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "minLength",
    "maxLength",
}
FILLER_CAPABILITIES = (
    ("Return the maintenance status of an unrelated synthetic service.", "service_id"),
    ("Look up the archival code of an unrelated laboratory sample.", "sample_id"),
    ("Retrieve the calibration date of an unrelated test instrument.", "instrument_id"),
    ("Return the storage tier of an unrelated backup volume.", "volume_id"),
)


def parse_args() -> argparse.Namespace:
    """解析项目目录与可复现实验参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def load_all_evaluation_ids(project_root: Path) -> set[str]:
    """读取全部既有评测来源，禁止方法训练污染诊断集和最终集。"""
    roots = [
        project_root / "data" / "evaluation" / "frozen_v1",
        project_root / "data" / "evaluation" / "final_holdout_v1",
        project_root / "data" / "evaluation" / "tool_world_v1",
        project_root / "data" / "evaluation" / "tool_world_final_v1",
    ]
    source_ids: set[str] = set()
    for root in roots:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"缺少评测清单：{manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = manifest.get("datasets")
        if not isinstance(datasets, dict):
            raise TypeError(f"评测清单 datasets 非对象：{manifest_path}")
        for filename in datasets:
            for row in read_jsonl(root / filename):
                source_id = row.get("source_id")
                if isinstance(source_id, str):
                    source_ids.add(source_id)
    return source_ids


def observed_arguments(base: dict[str, Any], target_name: str) -> dict[str, list[Any]]:
    """收集目标工具在所有标准调用中的候选参数值。"""
    observed: dict[str, list[Any]] = defaultdict(list)
    for call in base["expected_calls"]:
        if call["name"] != target_name:
            continue
        for key, candidates in call["arguments"].items():
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{target_name}.{key} 缺少候选标准值")
            observed[key].extend(copy.deepcopy(candidates))
    return dict(observed)


def synthetic_enum_value(
    schema_type: str, observed: list[Any], target_name: str, parameter_name: str, seed: int
) -> Any | None:
    """生成与全部标准值不等的同类型 enum 值。"""
    if schema_type == "string":
        value: Any = f"unsupported_{stable_digest(target_name, parameter_name, str(seed))[:8]}"
    elif schema_type == "boolean":
        boolean_values = [item for item in observed if isinstance(item, bool)]
        if not boolean_values:
            return None
        value = not boolean_values[0]
    elif schema_type == "integer":
        numeric = [
            item for item in observed if isinstance(item, int) and not isinstance(item, bool)
        ]
        if not numeric:
            return None
        value = max(numeric) + 7919
    elif schema_type == "number":
        numeric = [
            item
            for item in observed
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if not numeric:
            return None
        value = max(float(item) for item in numeric) + 7919.25
    else:
        return None
    return None if value in observed else value


def find_constraint(
    base: dict[str, Any], family: str, seed: int
) -> dict[str, Any] | None:
    """为单目标意图寻找指定类型、可验证且排除全部标准值的能力约束。"""
    call_names = {call["name"] for call in base["expected_calls"]}
    if len(call_names) != 1:
        return None
    target_name = next(iter(call_names))
    source_tool = next(
        (tool for tool in base["tools"] if tool["function"]["name"] == target_name), None
    )
    if source_tool is None:
        raise ValueError(f"标准答案调用了未声明工具：{target_name}")
    properties = source_tool["function"]["parameters"].get("properties", {})
    observed = observed_arguments(base, target_name)
    ordered_keys = sorted(
        observed, key=lambda key: stable_digest(str(seed), family, target_name, key)
    )
    for key in ordered_keys:
        schema_type = properties.get(key, {}).get("type")
        candidates = observed[key]
        if family == "enum" and schema_type in SCALAR_TYPES:
            value = synthetic_enum_value(schema_type, candidates, target_name, key, seed)
            if value is not None:
                return {
                    "family": family,
                    "target_name": target_name,
                    "parameter_name": key,
                    "schema_update": {"enum": [value]},
                    "description": f"This restricted interface only supports {value!r}.",
                    "gold_values": candidates,
                }
        if family == "numeric_range" and schema_type in {"integer", "number"}:
            numeric = [
                item
                for item in candidates
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            ]
            if not numeric:
                continue
            delta = 1 if schema_type == "integer" else 0.5
            maximum = min(float(item) for item in numeric) - delta
            if schema_type == "integer":
                maximum = int(maximum)
            return {
                "family": family,
                "target_name": target_name,
                "parameter_name": key,
                "schema_update": {"maximum": maximum},
                "description": f"This restricted interface only supports values at most {maximum!r}.",
                "gold_values": candidates,
            }
        if family == "string_pattern" and schema_type == "string":
            strings = [item for item in candidates if isinstance(item, str)]
            if not strings:
                continue
            token = f"supported_{stable_digest(target_name, key, str(seed))[:8]}"
            pattern = f"^{re.escape(token)}$"
            if any(re.search(pattern, item) for item in strings):
                continue
            return {
                "family": family,
                "target_name": target_name,
                "parameter_name": key,
                "schema_update": {"pattern": pattern},
                "description": f"This restricted interface only supports strings matching {pattern!r}.",
                "gold_values": candidates,
            }
    return None


def select_factorial_sources(
    pool: list[dict[str, Any]], seed: int, quotas: dict[str, int] | None = None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """按固定配额选择互异意图，并保持类别尽可能均衡。"""
    active_quotas = quotas or CONSTRAINT_QUOTAS
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_ids: set[str] = set()
    for family, quota in active_quotas.items():
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for base in pool:
            if base["source_id"] in used_ids:
                continue
            constraint = find_constraint(base, family, seed)
            if constraint is not None:
                candidates.append((base, constraint))
        candidates.sort(
            key=lambda item: stable_digest(
                str(seed), family, item[0]["category"], item[0]["source_id"]
            )
        )
        by_category: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for item in candidates:
            by_category[item[0]["category"]].append(item)
        family_selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        while len(family_selected) < quota:
            progressed = False
            for category in sorted(by_category):
                while by_category[category] and by_category[category][0][0]["source_id"] in used_ids:
                    by_category[category].pop(0)
                if by_category[category] and len(family_selected) < quota:
                    item = by_category[category].pop(0)
                    family_selected.append(item)
                    used_ids.add(item[0]["source_id"])
                    progressed = True
            if not progressed:
                break
        if len(family_selected) != quota:
            raise RuntimeError(
                f"约束族 {family} 仅找到 {len(family_selected)}/{quota} 个互异单目标意图"
            )
        selected.extend(family_selected)
    return selected


def renamed_tool(source_tool: dict[str, Any], name: str) -> dict[str, Any]:
    """复制工具并替换为不携带标签语义的稳定名称。"""
    tool = copy.deepcopy(source_tool)
    tool["function"]["name"] = name
    return tool


def constrained_tool(
    source_tool: dict[str, Any], name: str, constraint: dict[str, Any]
) -> dict[str, Any]:
    """依据显式 JSON Schema 规则构造能力不兼容工具。"""
    tool = renamed_tool(source_tool, name)
    function = tool["function"]
    function["description"] = (
        f"{function.get('description', '')} {constraint['description']}"
    ).strip()
    property_schema = function["parameters"]["properties"][constraint["parameter_name"]]
    for key in REMOVED_CONSTRAINT_KEYS:
        property_schema.pop(key, None)
    property_schema.update(copy.deepcopy(constraint["schema_update"]))
    property_schema["description"] = (
        f"{property_schema.get('description', '')} {constraint['description']}"
    ).strip()
    return tool


def filler_tool(source_id: str, slot: str, seed: int) -> dict[str, Any]:
    """构造无关控制工具，用于固定四象限的工具数量。"""
    digest = stable_digest(str(seed), source_id, slot)
    description, parameter_stem = FILLER_CAPABILITIES[int(digest[:8], 16) % len(FILLER_CAPABILITIES)]
    name = stable_alias("tool", f"{source_id}:filler:{slot}", seed)
    parameter_name = f"{parameter_stem}_{digest[8:14]}"
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    parameter_name: {
                        "type": "string",
                        "description": "Identifier required by this unrelated control capability.",
                    }
                },
                "required": [parameter_name],
                "additionalProperties": False,
            },
        },
    }


def shuffle_tools(tools: list[dict[str, Any]], source_id: str, quadrant: str, seed: int) -> None:
    """确定性打乱工具与参数顺序，避免位置泄露标签。"""
    rng = random.Random(int(stable_digest(str(seed), source_id, quadrant)[:16], 16))
    rng.shuffle(tools)
    for tool in tools:
        parameters = tool["function"]["parameters"]
        properties = list(parameters.get("properties", {}).items())
        rng.shuffle(properties)
        parameters["properties"] = dict(properties)
        required = parameters.get("required", [])
        rng.shuffle(required)


def build_factorial_worlds(
    base: dict[str, Any],
    constraint: dict[str, Any],
    seed: int,
    track: str = "fctw_train_v1",
    id_prefix: str = "fctw",
) -> list[dict[str, Any]]:
    """为同一意图构造工具数量一致的 E10、E01、E11、E00。"""
    target_name = constraint["target_name"]
    source_tool = next(
        tool for tool in base["tools"] if tool["function"]["name"] == target_name
    )
    other_tools = [
        copy.deepcopy(tool)
        for tool in base["tools"]
        if tool["function"]["name"] != target_name
    ]
    valid_name = stable_alias("tool", f"{base['source_id']}:capability:a", seed)
    near_name = stable_alias("tool", f"{base['source_id']}:capability:b", seed)
    valid = renamed_tool(source_tool, valid_name)
    near = constrained_tool(source_tool, near_name, constraint)
    filler_valid = filler_tool(base["source_id"], "a", seed)
    filler_near = filler_tool(base["source_id"], "b", seed)
    expected = copy.deepcopy(base["expected_calls"])
    for call in expected:
        call["name"] = valid_name

    slot_tools = {
        "E10": [valid, filler_near],
        "E01": [filler_valid, near],
        "E11": [valid, near],
        "E00": [filler_valid, filler_near],
    }
    worlds: list[dict[str, Any]] = []
    for quadrant, factors in QUADRANTS.items():
        tools = copy.deepcopy([*other_tools, *slot_tools[quadrant]])
        shuffle_tools(tools, base["source_id"], quadrant, seed)
        calls = copy.deepcopy(expected if factors["valid"] else [])
        worlds.append(
            {
                "id": f"{id_prefix}_{base['source_id']}__{quadrant}",
                "base_id": f"{id_prefix}_{base['source_id']}",
                "source_id": base["source_id"],
                "track": track,
                "category": base["category"],
                "world_type": quadrant,
                "valid_capability": factors["valid"],
                "near_miss_capability": factors["near_miss"],
                "expected_decision": factors["decision"],
                "messages": copy.deepcopy(base["messages"]),
                "tools": tools,
                "expected_calls": calls,
                "transformation": {
                    "seed": seed,
                    "constraint_family": constraint["family"],
                    "target_parameter": constraint["parameter_name"],
                    "constraint": copy.deepcopy(constraint["schema_update"]),
                    "valid_tool": valid_name if factors["valid"] else None,
                    "near_miss_tool": near_name if factors["near_miss"] else None,
                },
            }
        )
    validate_factorial_group(worlds, constraint)
    return worlds


def constraint_accepts(value: Any, constraint: dict[str, Any]) -> bool:
    """判断单个值是否满足本项目使用的受控约束。"""
    update = constraint["schema_update"]
    if "enum" in update:
        return value in update["enum"]
    if "maximum" in update:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value <= update["maximum"]
        )
    if "pattern" in update:
        return isinstance(value, str) and re.search(update["pattern"], value) is not None
    raise ValueError(f"未知约束：{update}")


def validate_factorial_group(
    worlds: list[dict[str, Any]], constraint: dict[str, Any]
) -> None:
    """验证四象限标签、工具数量、名称与能力约束均无矛盾。"""
    if {world["world_type"] for world in worlds} != set(QUADRANTS):
        raise RuntimeError("FCTW 四象限不完整")
    tool_counts = {len(world["tools"]) for world in worlds}
    if len(tool_counts) != 1:
        raise RuntimeError("FCTW 四象限工具数量不一致")
    for world in worlds:
        names = [tool["function"]["name"] for tool in world["tools"]]
        if len(names) != len(set(names)):
            raise RuntimeError(f"{world['id']} 存在重复工具名")
        expected_names = {call["name"] for call in world["expected_calls"]}
        if not expected_names.issubset(set(names)):
            raise RuntimeError(f"{world['id']} 的标准调用包含不可用工具")
        should_call = world["expected_decision"] == "call"
        if bool(world["expected_calls"]) != should_call:
            raise RuntimeError(f"{world['id']} 的 Call/Reject 标签与标准答案不一致")
        if any(not name.startswith("tool_") for name in expected_names):
            raise RuntimeError(f"{world['id']} 的目标工具名格式异常")
        near_name = world["transformation"]["near_miss_tool"]
        if near_name is not None and not near_name.startswith("tool_"):
            raise RuntimeError(f"{world['id']} 的 Near-miss 工具名格式异常")
    if any(constraint_accepts(value, constraint) for value in constraint["gold_values"]):
        raise RuntimeError("Near-miss 约束仍接受至少一个 Gold 值")


def register_dataset(project_root: Path) -> None:
    """在 LLaMA-Factory 数据清单中注册 FCTW 数据集。"""
    path = project_root / "data" / "dataset_info.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    additions = {
        "toolace_fctw_train": {
            "file_name": "sft/fctw_v1/train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "toolace_fctw_dev": {
            "file_name": "sft/fctw_v1/dev.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
    }
    for name, value in additions.items():
        if name in current and current[name] != value:
            raise RuntimeError(f"dataset_info 中 {name} 已存在但定义不同")
        current[name] = value
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """生成 100 个意图、400 个四象限世界并写入训练产物。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    excluded_ids = load_all_evaluation_ids(project_root)
    pool = build_base_tasks(project_root, CATEGORY_POOL_SIZES, excluded_ids, args.seed)
    selected = select_factorial_sources(pool, args.seed)
    worlds = [
        world
        for base, constraint in selected
        for world in build_factorial_worlds(base, constraint, args.seed)
    ]
    if len(worlds) != 400:
        raise RuntimeError(f"FCTW 世界数量应为 400，实际为 {len(worlds)}")
    if {world["source_id"] for world in worlds} & excluded_ids:
        raise RuntimeError("FCTW 训练来源与既有评测集发生泄漏")

    base_dir = project_root / "data" / "sft" / "failure_driven_v1"
    base_train_path = base_dir / "train.json"
    base_dev_path = base_dir / "dev.json"
    base_train = json.loads(base_train_path.read_text(encoding="utf-8"))
    base_dev = json.loads(base_dev_path.read_text(encoding="utf-8"))
    train = [*copy.deepcopy(base_train), *(world_to_sharegpt(world) for world in worlds)]
    stable_shuffle(train, args.seed, "fctw")

    output_dir = project_root / "data" / "sft" / "fctw_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    worlds_path = output_dir / "worlds.jsonl"
    if worlds_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{worlds_path}")
    write_jsonl(worlds_path, sorted(worlds, key=lambda row: row["id"]))
    data_metadata = {
        "train": write_dataset(output_dir / "train.json", train),
        "dev": write_dataset(output_dir / "dev.json", copy.deepcopy(base_dev)),
        "worlds": {"samples": len(worlds), "sha256": sha256_file(worlds_path)},
    }
    world_counts = Counter(world["world_type"] for world in worlds)
    constraint_counts = Counter(
        world["transformation"]["constraint_family"]
        for world in worlds
        if world["world_type"] == "E10"
    )
    category_counts = Counter(base["category"] for base, _ in selected)
    tool_count_mismatches = sum(
        len({len(world["tools"]) for world in worlds if world["base_id"] == base_id}) != 1
        for base_id in {world["base_id"] for world in worlds}
    )
    manifest = {
        "version": "fctw_sft_v1",
        "seed": args.seed,
        "method": "Factorial Counterfactual Tool-World SFT",
        "hypothesis": (
            "完整覆盖 Valid × Near-miss 四象限，可减少 E11 错误拒绝，"
            "同时保留 E01 的能力约束拒绝。"
        ),
        "base": {
            "dataset": "failure_driven_v1",
            "train_samples": len(base_train),
            "train_sha256": sha256_file(base_train_path),
            "dev_samples": len(base_dev),
            "dev_sha256": sha256_file(base_dev_path),
        },
        "controlled_budget": {
            "added_samples": len(worlds),
            "unique_added_intents": len(selected),
            "world_counts": dict(sorted(world_counts.items())),
            "call_samples": sum(bool(world["expected_calls"]) for world in worlds),
            "reject_samples": sum(not world["expected_calls"] for world in worlds),
            "constraint_counts": dict(sorted(constraint_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "shortcut_checks": {
            "tool_count_mismatched_groups": tool_count_mismatches,
            "opaque_name_prefix": "tool_",
            "position_randomized": True,
            "single_target_intent_only": True,
        },
        "data": data_metadata,
        "excluded_evaluation_source_ids": len(excluded_ids),
        "limitations": [
            "训练约束仅覆盖 enum 与数值上界；字符串模式约束保留为未见评测。",
            "标准 SFT 不显式加入因子化损失，方法收益只能归因于结构化数据覆盖。",
            "为获得无歧义 Reject 标签，仅使用全部标准调用属于同一目标工具的意图。",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    register_dataset(project_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
