#!/usr/bin/env python3
"""构建反事实工具世界诊断集。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from build_frozen_eval import convert_ground_truth, normalize_schema, read_jsonl, sha256_file, write_jsonl


DEFAULT_CATEGORY_SIZES = {
    "simple_python": 45,
    "multiple": 40,
    "parallel": 35,
    "parallel_multiple": 30,
}
SCALAR_TYPES = {"string", "integer", "number", "boolean"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建 Counterfactual Tool-World 诊断集")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--category-sizes",
        default=json.dumps(DEFAULT_CATEGORY_SIZES),
        help="各 BFCL 类别基础任务数的 JSON 对象",
    )
    return parser.parse_args()


def stable_digest(*parts: str) -> str:
    """生成跨运行稳定的十六进制摘要。"""
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def stable_alias(prefix: str, source: str, seed: int) -> str:
    """生成不携带原始语义的稳定接口别名。"""
    return f"{prefix}_{stable_digest(str(seed), prefix, source)[:10]}"


def load_excluded_ids(project_root: Path) -> set[str]:
    """汇总旧开发集和最终 holdout 已使用的 BFCL source ID。"""
    excluded: set[str] = set()
    roots = [
        project_root / "data" / "evaluation" / "frozen_v1",
        project_root / "data" / "evaluation" / "final_holdout_v1",
    ]
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for filename in manifest["datasets"]:
            for row in read_jsonl(root / filename):
                source_id = row.get("source_id")
                if isinstance(source_id, str):
                    excluded.add(source_id)
    return excluded


def function_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """建立工具名到函数定义的映射，并拒绝重复工具名。"""
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool["function"]
        if function["name"] in result:
            raise ValueError(f"工具名重复：{function['name']}")
        result[function["name"]] = function
    return result


def scalar_constraint(
    expected_calls: list[dict[str, Any]], tools: list[dict[str, Any]], target_name: str, seed: int
) -> tuple[str, Any] | None:
    """为目标工具寻找可确定性改写的标量参数约束。"""
    function = function_map(tools)[target_name]
    properties = function["parameters"].get("properties", {})
    observed_by_key: dict[str, list[Any]] = {}
    for call in expected_calls:
        if call["name"] != target_name:
            continue
        for key, candidates in call["arguments"].items():
            if not isinstance(candidates, list):
                raise TypeError(f"{target_name}.{key} 的候选值必须是列表")
            observed_by_key.setdefault(key, []).extend(candidates)

    for key in sorted(observed_by_key, key=lambda item: stable_digest(str(seed), target_name, item)):
        schema_type = properties.get(key, {}).get("type")
        observed = observed_by_key[key]
        if schema_type not in SCALAR_TYPES or not observed:
            continue
        if schema_type == "string":
            candidate: Any = f"unsupported_{stable_digest(target_name, key, str(seed))[:8]}"
        elif schema_type == "boolean":
            boolean_values = [value for value in observed if isinstance(value, bool)]
            if not boolean_values:
                continue
            candidate = not boolean_values[0]
            if candidate in observed:
                continue
        elif schema_type == "integer":
            numeric = [value for value in observed if isinstance(value, int) and not isinstance(value, bool)]
            if not numeric:
                continue
            candidate = max(numeric) + 7919
        else:
            numeric = [
                value
                for value in observed
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if not numeric:
                continue
            candidate = max(float(value) for value in numeric) + 7919.25
        if candidate not in observed:
            return key, candidate
    return None


def choose_target(
    expected_calls: list[dict[str, Any]], tools: list[dict[str, Any]], seed: int
) -> tuple[str, str, Any] | None:
    """选择能够构造约束替换的目标工具和参数。"""
    names = sorted(
        {call["name"] for call in expected_calls},
        key=lambda name: stable_digest(str(seed), "target", name),
    )
    schemas = function_map(tools)
    for name in names:
        if name not in schemas:
            raise ValueError(f"标准答案调用了未声明工具：{name}")
        constraint = scalar_constraint(expected_calls, tools, name, seed)
        if constraint is not None:
            return name, constraint[0], constraint[1]
    return None


def expected_calls_fit_tools(
    expected_calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> bool:
    """检查标准调用的工具名和参数名是否全部由当前 Schema 声明。"""
    schemas = function_map(tools)
    for call in expected_calls:
        function = schemas.get(call["name"])
        if function is None:
            return False
        properties = function["parameters"].get("properties", {})
        if set(call["arguments"]) - set(properties):
            return False
    return True


def identity_semantic_map(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """生成原始接口到规范语义接口的恒等映射。"""
    return {
        function["name"]: {
            "canonical_name": function["name"],
            "arguments": {
                key: key for key in function["parameters"].get("properties", {})
            },
        }
        for function in (tool["function"] for tool in tools)
    }


def make_world(
    base: dict[str, Any], world_type: str, tools: list[dict[str, Any]], expected_calls: list[dict[str, Any]],
    semantic_map: dict[str, Any], metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装单个工具世界样本。"""
    return {
        "id": f"tw_{base['source_id']}__{world_type}",
        "base_id": f"tw_{base['source_id']}",
        "source_id": base["source_id"],
        "track": "tool_world_v1",
        "category": base["category"],
        "world_type": world_type,
        "messages": copy.deepcopy(base["messages"]),
        "tools": tools,
        "expected_calls": expected_calls,
        "canonical_expected_calls": copy.deepcopy(expected_calls),
        "semantic_map": semantic_map,
        "transformation": metadata or {},
        "executable": False,
    }


def build_original_world(base: dict[str, Any]) -> dict[str, Any]:
    """构造未改写的原始工具世界。"""
    tools = copy.deepcopy(base["tools"])
    expected = copy.deepcopy(base["expected_calls"])
    return make_world(base, "original", tools, expected, identity_semantic_map(tools))


def build_ephemeral_world(base: dict[str, Any], seed: int) -> dict[str, Any]:
    """一致重命名工具和顶层参数，并随机排列接口顺序。"""
    tools = copy.deepcopy(base["tools"])
    expected = copy.deepcopy(base["expected_calls"])
    forward: dict[str, tuple[str, dict[str, str]]] = {}
    semantic_map: dict[str, Any] = {}
    for tool in tools:
        function = tool["function"]
        old_name = function["name"]
        new_name = stable_alias("tool", old_name, seed)
        properties = function["parameters"].get("properties", {})
        argument_map = {
            old_key: stable_alias("arg", f"{old_name}:{old_key}", seed) for old_key in properties
        }
        function["name"] = new_name
        function["parameters"]["properties"] = {
            argument_map[key]: value for key, value in properties.items()
        }
        function["parameters"]["required"] = [
            argument_map[key] for key in function["parameters"].get("required", [])
        ]
        forward[old_name] = (new_name, argument_map)
        semantic_map[new_name] = {
            "canonical_name": old_name,
            "arguments": {new: old for old, new in argument_map.items()},
        }

    for call in expected:
        new_name, argument_map = forward[call["name"]]
        call["name"] = new_name
        call["arguments"] = {argument_map[key]: value for key, value in call["arguments"].items()}

    rng = random.Random(int(stable_digest(str(seed), base["source_id"], "ephemeral")[:16], 16))
    rng.shuffle(tools)
    for tool in tools:
        parameters = tool["function"]["parameters"]
        properties = list(parameters.get("properties", {}).items())
        rng.shuffle(properties)
        parameters["properties"] = dict(properties)
        rng.shuffle(parameters.get("required", []))

    world = make_world(
        base,
        "ephemeral_interface",
        tools,
        expected,
        semantic_map,
        {"kind": "interface_preserving", "seed": seed},
    )
    world["canonical_expected_calls"] = copy.deepcopy(base["expected_calls"])
    return world


def build_capability_removal_world(base: dict[str, Any], target_name: str, seed: int) -> dict[str, Any]:
    """移除一个必要工具，并保留仍可独立完成的其余调用。"""
    tools = [
        copy.deepcopy(tool) for tool in base["tools"] if tool["function"]["name"] != target_name
    ]
    expected = [
        copy.deepcopy(call) for call in base["expected_calls"] if call["name"] != target_name
    ]
    return make_world(
        base,
        "capability_removal",
        tools,
        expected,
        identity_semantic_map(tools),
        {"kind": "capability_changing", "seed": seed, "removed_tool": target_name},
    )


def constrained_clone(
    source_tool: dict[str, Any], new_name: str, parameter_name: str, allowed_value: Any
) -> dict[str, Any]:
    """创建能力受限且名称不同的工具定义。"""
    clone = copy.deepcopy(source_tool)
    function = clone["function"]
    function["name"] = new_name
    function["description"] = (
        f"{function.get('description', '')} 此受限版本仅支持 Schema enum 中列出的值。"
    ).strip()
    property_schema = function["parameters"]["properties"][parameter_name]
    property_schema["enum"] = [allowed_value]
    property_schema["description"] = (
        f"{property_schema.get('description', '')} 当前仅支持 {allowed_value!r}。"
    ).strip()
    return clone


def build_substitution_world(
    base: dict[str, Any], target_name: str, parameter_name: str, allowed_value: Any, seed: int
) -> dict[str, Any]:
    """用参数能力不兼容的受限工具替换必要工具。"""
    replacement_name = stable_alias("limited", target_name, seed)
    tools: list[dict[str, Any]] = []
    for tool in base["tools"]:
        if tool["function"]["name"] == target_name:
            tools.append(constrained_clone(tool, replacement_name, parameter_name, allowed_value))
        else:
            tools.append(copy.deepcopy(tool))
    expected = [
        copy.deepcopy(call) for call in base["expected_calls"] if call["name"] != target_name
    ]
    semantic_map = identity_semantic_map(tools)
    return make_world(
        base,
        "capability_substitution",
        tools,
        expected,
        semantic_map,
        {
            "kind": "capability_changing",
            "seed": seed,
            "replaced_tool": target_name,
            "replacement_tool": replacement_name,
            "restricted_parameter": parameter_name,
            "allowed_value": allowed_value,
        },
    )


def build_distractor_world(
    base: dict[str, Any], target_name: str, parameter_name: str, allowed_value: Any, seed: int
) -> dict[str, Any]:
    """添加描述相似但参数能力不适用的困难干扰工具。"""
    distractor_name = stable_alias("distractor", target_name, seed)
    tools = copy.deepcopy(base["tools"])
    source_tool = next(tool for tool in base["tools"] if tool["function"]["name"] == target_name)
    tools.insert(0, constrained_clone(source_tool, distractor_name, parameter_name, allowed_value))
    world = make_world(
        base,
        "distractor",
        tools,
        copy.deepcopy(base["expected_calls"]),
        identity_semantic_map(tools),
        {
            "kind": "capability_changing",
            "seed": seed,
            "source_tool": target_name,
            "distractor_tool": distractor_name,
            "restricted_parameter": parameter_name,
            "allowed_value": allowed_value,
        },
    )
    return world


def validate_world(world: dict[str, Any]) -> None:
    """验证标准调用均受当前世界 Schema 约束。"""
    schemas = function_map(world["tools"])
    for call in world["expected_calls"]:
        if call["name"] not in schemas:
            raise ValueError(f"{world['id']} 的标准答案调用了不可用工具：{call['name']}")
        properties = schemas[call["name"]]["parameters"].get("properties", {})
        unknown = set(call["arguments"]) - set(properties)
        if unknown:
            raise ValueError(f"{world['id']} 的标准答案包含未知参数：{sorted(unknown)}")


def build_base_tasks(
    project_root: Path, category_sizes: dict[str, int], excluded_ids: set[str], seed: int
) -> list[dict[str, Any]]:
    """从未使用 BFCL source ID 中稳定选择可变换基础任务。"""
    data_root = (
        project_root
        / "vendor"
        / "gorilla"
        / "berkeley-function-call-leaderboard"
        / "bfcl_eval"
        / "data"
    )
    answer_root = data_root / "possible_answer"
    selected: list[dict[str, Any]] = []
    for category, size in category_sizes.items():
        if category == "irrelevance" or size <= 0:
            raise ValueError("诊断基础任务必须是正向工具调用样本，且类别数量必须为正数")
        source_rows = read_jsonl(data_root / f"BFCL_v4_{category}.json")
        answers = {
            row["id"]: row["ground_truth"]
            for row in read_jsonl(answer_root / f"BFCL_v4_{category}.json")
        }
        candidates = sorted(
            (row for row in source_rows if row["id"] not in excluded_ids),
            key=lambda row: stable_digest(str(seed), category, row["id"]),
        )
        category_selected = 0
        for row in candidates:
            turns = row["question"]
            if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], list):
                continue
            tools = [
                {"type": "function", "function": normalize_schema(function)}
                for function in row["function"]
            ]
            expected_calls = convert_ground_truth(answers[row["id"]])
            if not expected_calls_fit_tools(expected_calls, tools):
                continue
            target = choose_target(expected_calls, tools, seed)
            if target is None:
                continue
            selected.append(
                {
                    "source_id": row["id"],
                    "category": category,
                    "messages": turns[0],
                    "tools": tools,
                    "expected_calls": expected_calls,
                    "target": target,
                }
            )
            category_selected += 1
            if category_selected == size:
                break
        if category_selected != size:
            raise RuntimeError(f"类别 {category} 仅找到 {category_selected}/{size} 条可变换任务")
    return selected


def main() -> None:
    """构建五类工具世界、验证完整性并写入不可覆盖产物。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    category_sizes = json.loads(args.category_sizes)
    if not isinstance(category_sizes, dict) or not category_sizes:
        raise TypeError("category-sizes 必须是非空 JSON 对象")
    category_sizes = {str(key): int(value) for key, value in category_sizes.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "tool_world_v1.jsonl"
    manifest_path = output_dir / "manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"诊断集已存在，拒绝覆盖：{output_dir}")

    excluded_ids = load_excluded_ids(project_root)
    base_tasks = build_base_tasks(project_root, category_sizes, excluded_ids, args.seed)
    worlds: list[dict[str, Any]] = []
    for base in base_tasks:
        target_name, parameter_name, allowed_value = base.pop("target")
        worlds.extend(
            [
                build_original_world(base),
                build_ephemeral_world(base, args.seed),
                build_capability_removal_world(base, target_name, args.seed),
                build_substitution_world(
                    base, target_name, parameter_name, allowed_value, args.seed
                ),
                build_distractor_world(base, target_name, parameter_name, allowed_value, args.seed),
            ]
        )
    for world in worlds:
        validate_world(world)
    ids = [world["id"] for world in worlds]
    if len(ids) != len(set(ids)):
        raise RuntimeError("工具世界诊断集存在重复 ID")
    source_ids = {world["source_id"] for world in worlds}
    if source_ids & excluded_ids:
        raise RuntimeError("工具世界诊断集与旧评测集发生 source ID 泄漏")

    worlds.sort(key=lambda row: row["id"])
    write_jsonl(data_path, worlds)
    manifest = {
        "version": "tool_world_v1",
        "seed": args.seed,
        "policy": "同一基础请求的五个世界保持同组；不读取旧测试答案生成训练数据。",
        "base_tasks": len(base_tasks),
        "worlds_per_task": 5,
        "world_samples": len(worlds),
        "category_sizes": category_sizes,
        "excluded_source_ids": len(excluded_ids),
        "datasets": {
            data_path.name: {"samples": len(worlds), "sha256": sha256_file(data_path)}
        },
        "world_types": [
            "original",
            "ephemeral_interface",
            "capability_removal",
            "capability_substitution",
            "distractor",
        ],
        "limitations": [
            "能力替换通过收紧一个标量参数的 enum 构造，不代表所有现实工具能力变化。",
            "多调用任务允许继续调用仍可独立完成的工具，不把所有能力移除样本标为 no-tool。",
            "该诊断集用于方法选择，不能作为后续新方法的独立最终测试集。",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
