#!/usr/bin/env python3
"""依据工具世界诊断失败率构建能力加权反事实 SFT 数据。"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_counterfactual_sft import (
    CATEGORY_POOL_SIZES,
    load_all_evaluation_ids,
    stable_shuffle,
    transform_world,
    world_to_sharegpt,
    write_dataset,
)
from build_frozen_eval import sha256_file
from build_tool_world_eval import build_base_tasks, build_original_world


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建失败加权工具世界 SFT 数据")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    return parser.parse_args()


def register_dataset(project_root: Path) -> None:
    """在 LLaMA-Factory 数据清单中注册失败加权数据集。"""
    path = project_root / "data" / "dataset_info.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    additions = {
        "toolace_failure_weighted_train": {
            "file_name": "sft/failure_weighted_v1/train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "toolace_failure_weighted_dev": {
            "file_name": "sft/failure_weighted_v1/dev.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
    }
    for name, value in additions.items():
        if name in current and current[name] != value:
            raise RuntimeError(f"dataset_info 中 {name} 已存在但定义不同")
        current[name] = value
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_weighted_worlds(pool: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """用固定 400 条预算构造失败加权的工具世界监督。"""
    ordered = sorted(pool, key=lambda row: row["source_id"])
    if len(ordered) < 250:
        raise ValueError("失败加权数据至少需要 250 个互异基础意图")
    substitution_sources = ordered[:150]
    remaining = ordered[150:250]
    worlds = [build_original_world(base) for base in substitution_sources]
    worlds.extend(
        transform_world(base, "capability_substitution", seed)
        for base in substitution_sources
    )
    worlds.extend(
        transform_world(base, "capability_removal", seed) for base in remaining[:50]
    )
    worlds.extend(
        transform_world(base, "ephemeral_interface", seed) for base in remaining[50:75]
    )
    worlds.extend(
        transform_world(base, "distractor", seed) for base in remaining[75:100]
    )
    if len(worlds) != 400:
        raise RuntimeError("失败加权工具世界数量不为 400")
    return worlds


def main() -> None:
    """构建并记录失败加权训练集。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    excluded_ids = load_all_evaluation_ids(project_root)
    pool = build_base_tasks(project_root, CATEGORY_POOL_SIZES, excluded_ids, args.seed)
    worlds = build_weighted_worlds(pool, args.seed)

    base_dir = project_root / "data" / "sft" / "failure_driven_v1"
    base_train_path = base_dir / "train.json"
    base_dev_path = base_dir / "dev.json"
    base_train = json.loads(base_train_path.read_text(encoding="utf-8"))
    base_dev = json.loads(base_dev_path.read_text(encoding="utf-8"))
    train = [*copy.deepcopy(base_train), *(world_to_sharegpt(world) for world in worlds)]
    stable_shuffle(train, args.seed, "failure-weighted")
    output_dir = project_root / "data" / "sft" / "failure_weighted_v1"
    data_metadata = {
        "train": write_dataset(output_dir / "train.json", train),
        "dev": write_dataset(output_dir / "dev.json", copy.deepcopy(base_dev)),
    }
    world_counts = Counter(world["world_type"] for world in worlds)
    source_ids = {world["source_id"] for world in worlds}
    if source_ids & excluded_ids:
        raise RuntimeError("失败加权训练数据与评测 source ID 发生泄漏")
    manifest = {
        "version": "failure_weighted_tool_world_sft_v1",
        "seed": args.seed,
        "method": "Failure-Weighted Counterfactual Tool-World SFT",
        "hypothesis": "配对本身不足以产生跨世界约束学习，需要提高诊断最差能力变换的监督密度。",
        "base": {
            "dataset": "failure_driven_v1",
            "train_samples": len(base_train),
            "train_sha256": sha256_file(base_train_path),
            "dev_samples": len(base_dev),
            "dev_sha256": sha256_file(base_dev_path),
        },
        "controlled_budget": {
            "added_samples": len(worlds),
            "unique_added_intents": len(source_ids),
            "world_counts": dict(sorted(world_counts.items())),
        },
        "data": data_metadata,
        "excluded_evaluation_source_ids": len(excluded_ids),
        "limitations": [
            "该方法改变了工具世界类型分配，改进只能归因于失败加权数据策略，不能归因于新优化目标。",
            "能力替换仍由标量 enum 约束构造，尚未覆盖复杂组合能力变化。",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    register_dataset(project_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
