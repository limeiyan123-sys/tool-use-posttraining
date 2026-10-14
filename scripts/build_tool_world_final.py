#!/usr/bin/env python3
"""构建与 Stage 4 训练及方法选择集完全隔离的最终工具世界测试集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_counterfactual_sft import CATEGORY_POOL_SIZES, load_all_evaluation_ids
from build_frozen_eval import sha256_file, write_jsonl
from build_tool_world_eval import (
    build_base_tasks,
    build_capability_removal_world,
    build_distractor_world,
    build_ephemeral_world,
    build_original_world,
    build_substitution_world,
    validate_world,
)


DEFAULT_FINAL_CATEGORY_SIZES = {
    "simple_python": 28,
    "multiple": 2,
    "parallel": 15,
    "parallel_multiple": 15,
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建独立最终工具世界测试集")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--category-sizes",
        default=json.dumps(DEFAULT_FINAL_CATEGORY_SIZES),
        help="各 BFCL 类别基础任务数的 JSON 对象",
    )
    return parser.parse_args()


def source_id_digest(source_ids: set[str]) -> str:
    """计算来源 ID 集合的稳定摘要，便于审计数据边界。"""
    payload = "\n".join(sorted(source_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reconstruct_stage4_training_ids(project_root: Path) -> dict[str, set[str]]:
    """按冻结脚本和种子重建 Stage 4 使用的 BFCL 来源 ID。"""
    evaluation_ids = load_all_evaluation_ids(project_root)
    randomization_pool = build_base_tasks(
        project_root, CATEGORY_POOL_SIZES, evaluation_ids, seed=20261017
    )
    failure_pool = build_base_tasks(
        project_root, CATEGORY_POOL_SIZES, evaluation_ids, seed=20261018
    )
    randomization_ids = {row["source_id"] for row in randomization_pool}
    failure_weighted_ids = {
        row["source_id"]
        for row in sorted(failure_pool, key=lambda item: item["source_id"])[:250]
    }
    if len(randomization_ids) != 400 or len(failure_weighted_ids) != 250:
        raise RuntimeError("Stage 4 训练来源 ID 重建数量异常")
    return {
        "evaluation": evaluation_ids,
        "world_randomization_and_paired": randomization_ids,
        "failure_weighted": failure_weighted_ids,
    }


def build_worlds(base_tasks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """为每个基础任务构造五种成组工具世界。"""
    worlds: list[dict[str, Any]] = []
    for base in base_tasks:
        target_name, parameter_name, allowed_value = base.pop("target")
        current = [
            build_original_world(base),
            build_ephemeral_world(base, seed),
            build_capability_removal_world(base, target_name, seed),
            build_substitution_world(base, target_name, parameter_name, allowed_value, seed),
            build_distractor_world(base, target_name, parameter_name, allowed_value, seed),
        ]
        for world in current:
            world["track"] = "tool_world_final_v1"
            validate_world(world)
        worlds.extend(current)
    return worlds


def main() -> None:
    """生成一次性使用的独立最终测试集并记录完整隔离证据。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    category_sizes_raw = json.loads(args.category_sizes)
    if not isinstance(category_sizes_raw, dict) or not category_sizes_raw:
        raise TypeError("category-sizes 必须是非空 JSON 对象")
    category_sizes = {str(key): int(value) for key, value in category_sizes_raw.items()}
    if any(value <= 0 for value in category_sizes.values()):
        raise ValueError("最终测试集各类别数量必须为正数")

    data_path = output_dir / "tool_world_final_v1.jsonl"
    manifest_path = output_dir / "manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"最终测试集已存在，拒绝覆盖：{output_dir}")

    reconstructed = reconstruct_stage4_training_ids(project_root)
    excluded_ids = set().union(*reconstructed.values())
    base_tasks = build_base_tasks(project_root, category_sizes, excluded_ids, args.seed)
    final_ids = {row["source_id"] for row in base_tasks}
    if final_ids & excluded_ids:
        raise RuntimeError("独立最终测试集与训练集或既有评测集发生 source ID 泄漏")

    worlds = build_worlds(base_tasks, args.seed)
    worlds.sort(key=lambda row: row["id"])
    ids = [row["id"] for row in worlds]
    if len(ids) != len(set(ids)):
        raise RuntimeError("独立最终测试集存在重复样本 ID")

    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(data_path, worlds)
    manifest = {
        "version": "tool_world_final_v1",
        "seed": args.seed,
        "status": "方法与超参数冻结后一次性评测，不再据此修改方法。",
        "base_tasks": len(base_tasks),
        "worlds_per_task": 5,
        "world_samples": len(worlds),
        "category_sizes": category_sizes,
        "datasets": {
            data_path.name: {"samples": len(worlds), "sha256": sha256_file(data_path)}
        },
        "source_id_audit": {
            name: {"count": len(values), "sha256": source_id_digest(values)}
            for name, values in sorted(reconstructed.items())
        },
        "all_excluded_source_ids": {
            "count": len(excluded_ids),
            "sha256": source_id_digest(excluded_ids),
        },
        "final_source_ids": {
            "count": len(final_ids),
            "sha256": source_id_digest(final_ids),
        },
        "leakage_check": "最终 source ID 与旧评测、Tool-World 方法选择集及全部 Stage 4 BFCL 训练来源完全互斥。",
        "limitations": [
            "最终集规模为 60 个基础任务、300 个工具世界，置信区间仍可能较宽。",
            "严格排除训练与既有评测来源后，multiple 类仅剩 2 条可变换任务，类别分布不均衡。",
            "能力替换仍只覆盖单个标量参数的 enum 收紧。",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
