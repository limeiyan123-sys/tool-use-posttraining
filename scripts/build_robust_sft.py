#!/usr/bin/env python3
"""从已清洗训练集构造 Schema 鲁棒与失败驱动增强数据。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]{3,}")
STOPWORDS = {
    "and", "are", "can", "for", "from", "get", "how", "need", "please", "the", "this",
    "tool", "use", "what", "with", "you", "your",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构造失败驱动的 Schema 鲁棒 SFT 数据")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schema-count", type=int, default=400)
    parser.add_argument("--hard-negative-count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20261014)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(sample: dict[str, Any], namespace: str) -> str:
    """生成与输入顺序无关的稳定排序键。"""
    payload = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{namespace}\n{payload}".encode("utf-8")).hexdigest()


def alias_name(kind: str, original: str, salt: str) -> str:
    """生成不携带原词义、但满足常见函数命名规则的稳定别名。"""
    digest = hashlib.sha256(f"{salt}\n{kind}\n{original}".encode("utf-8")).hexdigest()[:10]
    return f"{kind}_{digest}"


def parse_tools(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """严格解析样本工具列表。"""
    tools = json.loads(sample["tools"])
    if not isinstance(tools, list):
        raise ValueError("工具列表必须是 JSON 数组")
    return tools


def function_calls(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """严格解析对话中的函数调用。"""
    calls: list[dict[str, Any]] = []
    for message in sample["conversations"]:
        if message["from"] != "function_call":
            continue
        value = json.loads(message["value"])
        current_calls = value if isinstance(value, list) else [value]
        if not current_calls:
            raise ValueError("并行 function_call 列表不得为空")
        for call in current_calls:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ValueError("function_call 必须且只能包含 name、arguments")
            if not isinstance(call["arguments"], dict):
                raise TypeError("function_call.arguments 必须是对象")
            calls.append(call)
    return calls


def permute_schema(tools: list[dict[str, Any]], rng: random.Random) -> None:
    """打乱工具、参数与 required 顺序，但不改变 Schema 语义。"""
    rng.shuffle(tools)
    for tool in tools:
        parameters = tool["parameters"]
        properties = list(parameters.get("properties", {}).items())
        rng.shuffle(properties)
        parameters["properties"] = dict(properties)
        if "required" in parameters:
            rng.shuffle(parameters["required"])


def schema_alias_sample(sample: dict[str, Any], salt: str, seed: int) -> dict[str, Any]:
    """一致重命名工具和顶层参数，构造未见 Schema 训练样本。"""
    transformed = copy.deepcopy(sample)
    tools = parse_tools(transformed)
    tool_maps: dict[str, tuple[str, dict[str, str]]] = {}
    for tool in tools:
        old_tool_name = tool["name"]
        new_tool_name = alias_name("tool", old_tool_name, salt)
        properties = tool["parameters"].get("properties", {})
        parameter_map = {
            name: alias_name("arg", f"{old_tool_name}\n{name}", salt) for name in properties
        }
        tool["name"] = new_tool_name
        tool["parameters"]["properties"] = {
            parameter_map[name]: schema for name, schema in properties.items()
        }
        tool["parameters"]["required"] = [
            parameter_map[name] for name in tool["parameters"].get("required", [])
        ]
        tool_maps[old_tool_name] = (new_tool_name, parameter_map)

    for message in transformed["conversations"]:
        if message["from"] != "function_call":
            continue
        value = json.loads(message["value"])
        is_parallel = isinstance(value, list)
        current_calls = value if is_parallel else [value]
        renamed_calls: list[dict[str, Any]] = []
        for call in current_calls:
            if call["name"] not in tool_maps:
                raise ValueError(f"调用了未声明工具：{call['name']}")
            new_name, parameter_map = tool_maps[call["name"]]
            unknown = set(call["arguments"]) - set(parameter_map)
            if unknown:
                raise ValueError(f"调用包含未声明参数：{sorted(unknown)}")
            renamed_calls.append(
                {
                    "name": new_name,
                    "arguments": {
                        parameter_map[name]: argument for name, argument in call["arguments"].items()
                    },
                }
            )
        message["value"] = json.dumps(
            renamed_calls if is_parallel else renamed_calls[0], ensure_ascii=False, sort_keys=True
        )

    rng = random.Random(int(stable_key(sample, f"alias-{seed}-{salt}")[:16], 16))
    permute_schema(tools, rng)
    transformed["tools"] = json.dumps(tools, ensure_ascii=False, sort_keys=False)
    return transformed


def permuted_sample(sample: dict[str, Any], namespace: str, seed: int) -> dict[str, Any]:
    """生成仅改变 Schema 顺序的语义等价样本。"""
    transformed = copy.deepcopy(sample)
    tools = parse_tools(transformed)
    rng = random.Random(int(stable_key(sample, f"{namespace}-{seed}")[:16], 16))
    permute_schema(tools, rng)
    transformed["tools"] = json.dumps(tools, ensure_ascii=False, sort_keys=False)
    return transformed


def lexical_tokens(text: str) -> set[str]:
    """提取用于困难负例排序的轻量词元。"""
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if token.lower() not in STOPWORDS}


def no_tool_difficulty(sample: dict[str, Any]) -> float:
    """用用户请求与候选工具描述的最大词元重合度估计负例难度。"""
    user_text = " ".join(
        message["value"] for message in sample["conversations"] if message["from"] == "human"
    )
    user_tokens = lexical_tokens(user_text)
    if not user_tokens:
        return 0.0
    scores: list[float] = []
    for tool in parse_tools(sample):
        schema_text = " ".join(
            [tool.get("name", ""), tool.get("description", "")]
            + [value.get("description", "") for value in tool["parameters"].get("properties", {}).values()]
        )
        tool_tokens = lexical_tokens(schema_text)
        scores.append(len(user_tokens & tool_tokens) / max(1, len(user_tokens)))
    return max(scores, default=0.0)


def validate_sample(sample: dict[str, Any]) -> None:
    """验证每个函数调用均受变换后 Schema 约束。"""
    tools = parse_tools(sample)
    schemas = {tool["name"]: tool["parameters"] for tool in tools}
    if len(schemas) != len(tools):
        raise ValueError("存在重复工具名")
    for call in function_calls(sample):
        if call["name"] not in schemas:
            raise ValueError(f"调用了未声明工具：{call['name']}")
        properties = schemas[call["name"]].get("properties", {})
        unknown = set(call["arguments"]) - set(properties)
        missing = set(schemas[call["name"]].get("required", [])) - set(call["arguments"])
        if unknown or missing:
            raise ValueError(f"Schema 不一致：unknown={sorted(unknown)}, missing={sorted(missing)}")


def deduplicate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按完整序列化内容去重，同时保持首次出现顺序。"""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        key = stable_key(sample, "deduplicate")
        if key not in seen:
            seen.add(key)
            result.append(sample)
    return result


def write_dataset(path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """保存数据集并返回可追溯元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"samples": len(samples), "sha256": sha256_file(path)}


def main() -> None:
    """构造 Schema-only 消融集和完整失败驱动增强集。"""
    args = parse_args()
    train = json.loads(args.train.read_text(encoding="utf-8"))
    dev = json.loads(args.dev.read_text(encoding="utf-8"))
    if args.schema_count <= 0 or args.schema_count > len(train):
        raise ValueError("schema-count 超出训练集范围")

    ranked = sorted(
        train,
        key=lambda sample: (
            -(5 * len(function_calls(sample)) + len(parse_tools(sample))),
            stable_key(sample, "schema-rank"),
        ),
    )
    schema_sources = [sample for sample in ranked if function_calls(sample)][: args.schema_count]
    schema_augmented = [
        schema_alias_sample(sample, f"schema-{index}", args.seed)
        for index, sample in enumerate(schema_sources)
    ]

    no_tool = [sample for sample in train if not function_calls(sample)]
    hard_negative_sources = sorted(
        no_tool,
        key=lambda sample: (-no_tool_difficulty(sample), stable_key(sample, "negative-rank")),
    )[: args.hard_negative_count]
    hard_negatives = [
        permuted_sample(sample, f"hard-negative-{index}", args.seed)
        for index, sample in enumerate(hard_negative_sources)
    ]

    multi_call_sources = [sample for sample in train if len(function_calls(sample)) > 1]
    multi_call_augmented = [
        schema_alias_sample(sample, f"multi-call-{index}", args.seed)
        for index, sample in enumerate(sorted(multi_call_sources, key=lambda x: stable_key(x, "multi")))
    ]

    schema_only = deduplicate([*train, *schema_augmented])
    full = deduplicate([*train, *schema_augmented, *hard_negatives, *multi_call_augmented])
    for sample in [*schema_only, *full, *dev]:
        validate_sample(sample)

    schema_dir = args.output_root / "schema_only_v1"
    full_dir = args.output_root / "failure_driven_v1"
    manifest = {
        "seed": args.seed,
        "source": {
            "train": {"path": str(args.train), "samples": len(train), "sha256": sha256_file(args.train)},
            "dev": {"path": str(args.dev), "samples": len(dev), "sha256": sha256_file(args.dev)},
        },
        "selection": {
            "schema_alias": len(schema_augmented),
            "hard_no_tool": len(hard_negatives),
            "multi_call": len(multi_call_augmented),
        },
        "schema_only": {
            "train": write_dataset(schema_dir / "train.json", schema_only),
            "dev": write_dataset(schema_dir / "dev.json", dev),
        },
        "failure_driven": {
            "train": write_dataset(full_dir / "train.json", full),
            "dev": write_dataset(full_dir / "dev.json", dev),
        },
    }
    for directory in (schema_dir, full_dir):
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
