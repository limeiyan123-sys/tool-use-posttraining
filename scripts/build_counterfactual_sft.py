#!/usr/bin/env python3
"""构建等预算的工具世界随机增强与反事实配对 SFT 数据。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_frozen_eval import read_jsonl, sha256_file
from build_tool_world_eval import (
    build_base_tasks,
    build_capability_removal_world,
    build_distractor_world,
    build_ephemeral_world,
    build_original_world,
    build_substitution_world,
)


CATEGORY_POOL_SIZES = {
    "simple_python": 100,
    "multiple": 100,
    "parallel": 100,
    "parallel_multiple": 100,
}
TRANSFORM_TYPES = (
    "ephemeral_interface",
    "capability_removal",
    "capability_substitution",
    "distractor",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建工具世界后训练数据")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    return parser.parse_args()


def load_all_evaluation_ids(project_root: Path) -> set[str]:
    """读取所有既有评测集 source ID，避免训练数据污染。"""
    roots = [
        project_root / "data" / "evaluation" / "frozen_v1",
        project_root / "data" / "evaluation" / "final_holdout_v1",
        project_root / "data" / "evaluation" / "tool_world_v1",
    ]
    source_ids: set[str] = set()
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for filename in manifest["datasets"]:
            for row in read_jsonl(root / filename):
                source_id = row.get("source_id")
                if isinstance(source_id, str):
                    source_ids.add(source_id)
    return source_ids


def materialize_arguments(arguments: dict[str, list[Any]]) -> dict[str, Any]:
    """从 BFCL 候选标准值中确定性选择第一个合法值。"""
    result: dict[str, Any] = {}
    for key, candidates in arguments.items():
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"参数 {key} 没有可用候选值")
        result[key] = copy.deepcopy(candidates[0])
    return result


def world_to_sharegpt(world: dict[str, Any]) -> dict[str, Any]:
    """将工具世界样本转换为 LLaMA-Factory ShareGPT 工具格式。"""
    user_messages = [
        message["content"] for message in world["messages"] if message.get("role") == "user"
    ]
    if len(user_messages) != 1:
        raise ValueError(f"{world['id']} 不是单轮单用户消息")
    calls = [
        {"name": call["name"], "arguments": materialize_arguments(call["arguments"])}
        for call in world["expected_calls"]
    ]
    conversations = [{"from": "human", "value": user_messages[0]}]
    if calls:
        value: dict[str, Any] | list[dict[str, Any]] = calls[0] if len(calls) == 1 else calls
        conversations.append(
            {
                "from": "function_call",
                "value": json.dumps(value, ensure_ascii=False, sort_keys=True),
            }
        )
    else:
        conversations.append(
            {
                "from": "gpt",
                "value": "The current tool environment cannot complete this request.",
            }
        )
    tools = [copy.deepcopy(tool["function"]) for tool in world["tools"]]
    return {
        "conversations": conversations,
        "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
    }


def transform_world(base: dict[str, Any], transform_type: str, seed: int) -> dict[str, Any]:
    """依据统一变换类型构造训练工具世界。"""
    target_name, parameter_name, allowed_value = base["target"]
    if transform_type == "ephemeral_interface":
        return build_ephemeral_world(base, seed)
    if transform_type == "capability_removal":
        return build_capability_removal_world(base, target_name, seed)
    if transform_type == "capability_substitution":
        return build_substitution_world(base, target_name, parameter_name, allowed_value, seed)
    if transform_type == "distractor":
        return build_distractor_world(base, target_name, parameter_name, allowed_value, seed)
    raise ValueError(f"未知工具世界变换：{transform_type}")


def stable_shuffle(samples: list[dict[str, Any]], seed: int, namespace: str) -> None:
    """以固定种子打乱训练样本，避免来源块状排列。"""
    rng_seed = int(hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).hexdigest()[:16], 16)
    random.Random(rng_seed).shuffle(samples)


def write_dataset(path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """不可覆盖地写入训练数据并返回校验信息。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"samples": len(samples), "sha256": sha256_file(path)}


def update_dataset_info(project_root: Path) -> None:
    """注册两个新数据集，已有同名配置必须完全一致。"""
    path = project_root / "data" / "dataset_info.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    additions = {
        "toolace_world_randomization_train": {
            "file_name": "sft/world_randomization_v1/train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "toolace_world_randomization_dev": {
            "file_name": "sft/world_randomization_v1/dev.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "toolace_counterfactual_paired_train": {
            "file_name": "sft/counterfactual_paired_v1/train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "toolace_counterfactual_paired_dev": {
            "file_name": "sft/counterfactual_paired_v1/dev.json",
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
    """生成同分布、等样本预算的随机增强与反事实配对数据。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    excluded_ids = load_all_evaluation_ids(project_root)
    pool = build_base_tasks(project_root, CATEGORY_POOL_SIZES, excluded_ids, args.seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in pool:
        by_category[sample["category"]].append(sample)

    paired_sources: list[dict[str, Any]] = []
    unpaired_sources: list[dict[str, Any]] = []
    for category in CATEGORY_POOL_SIZES:
        if len(by_category[category]) != 100:
            raise RuntimeError(f"类别 {category} 的训练候选数量不为 100")
        paired_sources.extend(by_category[category][:50])
        unpaired_sources.extend(by_category[category][50:100])
    if {row["source_id"] for row in paired_sources} & {
        row["source_id"] for row in unpaired_sources
    }:
        raise RuntimeError("配对与非配对训练来源发生重叠")

    paired_sources.sort(key=lambda row: row["source_id"])
    unpaired_sources.sort(key=lambda row: row["source_id"])
    assignments = [TRANSFORM_TYPES[index % len(TRANSFORM_TYPES)] for index in range(200)]
    transform_counts = Counter(assignments)
    paired_worlds = [build_original_world(base) for base in paired_sources]
    paired_worlds.extend(
        transform_world(base, transform_type, args.seed)
        for base, transform_type in zip(paired_sources, assignments)
    )
    randomized_worlds = [build_original_world(base) for base in paired_sources]
    randomized_worlds.extend(
        transform_world(base, transform_type, args.seed)
        for base, transform_type in zip(unpaired_sources, assignments)
    )

    base_train_path = project_root / "data" / "sft" / "failure_driven_v1" / "train.json"
    base_dev_path = project_root / "data" / "sft" / "failure_driven_v1" / "dev.json"
    base_train = json.loads(base_train_path.read_text(encoding="utf-8"))
    base_dev = json.loads(base_dev_path.read_text(encoding="utf-8"))
    paired_train = [*copy.deepcopy(base_train), *(world_to_sharegpt(world) for world in paired_worlds)]
    randomized_train = [
        *copy.deepcopy(base_train),
        *(world_to_sharegpt(world) for world in randomized_worlds),
    ]
    stable_shuffle(paired_train, args.seed, "counterfactual-paired")
    stable_shuffle(randomized_train, args.seed, "world-randomization")
    if len(paired_train) != len(randomized_train):
        raise RuntimeError("两个训练对照的数据量不一致")

    random_dir = project_root / "data" / "sft" / "world_randomization_v1"
    paired_dir = project_root / "data" / "sft" / "counterfactual_paired_v1"
    random_metadata = {
        "train": write_dataset(random_dir / "train.json", randomized_train),
        "dev": write_dataset(random_dir / "dev.json", copy.deepcopy(base_dev)),
    }
    paired_metadata = {
        "train": write_dataset(paired_dir / "train.json", paired_train),
        "dev": write_dataset(paired_dir / "dev.json", copy.deepcopy(base_dev)),
    }
    manifest = {
        "version": "counterfactual_tool_world_sft_v1",
        "seed": args.seed,
        "base": {
            "dataset": "failure_driven_v1",
            "train_samples": len(base_train),
            "train_sha256": sha256_file(base_train_path),
            "dev_samples": len(base_dev),
            "dev_sha256": sha256_file(base_dev_path),
        },
        "controlled_budget": {
            "added_samples": 400,
            "original_worlds": 200,
            "transformed_worlds": 200,
            "transform_counts": dict(sorted(transform_counts.items())),
        },
        "world_randomization": {
            "policy": "原始世界与变换世界来自互不重复的意图。",
            "unique_added_intents": 400,
            "data": random_metadata,
        },
        "counterfactual_paired": {
            "policy": "每个意图同时提供原始世界与一个变换世界。",
            "unique_added_intents": 200,
            "data": paired_metadata,
        },
        "excluded_evaluation_source_ids": len(excluded_ids),
        "leakage_check": "训练 source ID 与 frozen_v1、final_holdout_v1、tool_world_v1 完全互斥。",
        "limitations": [
            "两个数据集的样本数和变换类型数量一致，但独立意图覆盖数有意不同。",
            "标准 SFT 不显式计算跨世界损失；本实验检验重复意图下的反事实监督是否有额外价值。",
        ],
    }
    for directory in (random_dir, paired_dir):
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    update_dataset_info(project_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
