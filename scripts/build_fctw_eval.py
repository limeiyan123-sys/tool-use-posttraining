#!/usr/bin/env python3
"""构建 FCTW 的独立四象限与未见能力约束评测集。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_fctw_sft import (
    build_factorial_worlds,
    load_all_evaluation_ids,
    select_factorial_sources,
)
from build_frozen_eval import read_jsonl, sha256_file, write_jsonl
from build_tool_world_eval import build_base_tasks


EVAL_CATEGORY_POOL_SIZES = {
    "simple_python": 80,
    "multiple": 70,
    "parallel": 80,
    "parallel_multiple": 70,
}
EVAL_CONSTRAINT_QUOTAS = {
    "enum": 10,
    "numeric_range": 10,
    "string_pattern": 10,
}


def parse_args() -> argparse.Namespace:
    """解析评测集输出目录和随机种子。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    return parser.parse_args()


def load_fctw_training_ids(project_root: Path) -> set[str]:
    """读取 FCTW 训练来源，防止同一意图进入独立评测。"""
    path = project_root / "data" / "sft" / "fctw_v1" / "worlds.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 FCTW 训练世界：{path}")
    return {row["source_id"] for row in read_jsonl(path)}


def validate_eval_worlds(worlds: list[dict[str, Any]], excluded_ids: set[str]) -> None:
    """验证评测四象限完整、互斥且未见约束只用于测试。"""
    source_ids = {world["source_id"] for world in worlds}
    if source_ids & excluded_ids:
        raise RuntimeError("FCTW 评测来源与训练或既有评测来源重叠")
    by_base: dict[str, set[str]] = {}
    for world in worlds:
        by_base.setdefault(world["base_id"], set()).add(world["world_type"])
    expected = {"E10", "E01", "E11", "E00"}
    incomplete = [base_id for base_id, quadrants in by_base.items() if quadrants != expected]
    if incomplete:
        raise RuntimeError(f"FCTW 评测四象限不完整：{incomplete}")


def main() -> None:
    """生成 30 个未见意图、120 个严格四象限评测样本。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    excluded_ids = load_all_evaluation_ids(project_root) | load_fctw_training_ids(project_root)
    pool = build_base_tasks(
        project_root, EVAL_CATEGORY_POOL_SIZES, excluded_ids, args.seed
    )
    selected = select_factorial_sources(
        pool, args.seed, quotas=EVAL_CONSTRAINT_QUOTAS
    )
    worlds = [
        world
        for base, constraint in selected
        for world in build_factorial_worlds(
            base,
            constraint,
            args.seed,
            track="fctw_eval_v1",
            id_prefix="fctw_eval",
        )
    ]
    validate_eval_worlds(worlds, excluded_ids)
    worlds.sort(key=lambda row: row["id"])
    output_dir.mkdir(parents=True)
    data_path = output_dir / "fctw_eval_v1.jsonl"
    write_jsonl(data_path, worlds)
    constraint_counts = Counter(
        world["transformation"]["constraint_family"]
        for world in worlds
        if world["world_type"] == "E10"
    )
    category_counts = Counter(base["category"] for base, _ in selected)
    manifest: dict[str, Any] = {
        "version": "fctw_eval_v1",
        "seed": args.seed,
        "policy": (
            "独立来源意图的 Valid × Near-miss 四象限；string_pattern 仅用于未见约束测试。"
        ),
        "base_tasks": len(selected),
        "worlds_per_task": 4,
        "world_samples": len(worlds),
        "constraint_counts": dict(sorted(constraint_counts.items())),
        "unseen_constraint_family": "string_pattern",
        "category_counts": dict(sorted(category_counts.items())),
        "excluded_source_ids": len(excluded_ids),
        "datasets": {
            data_path.name: {"samples": len(worlds), "sha256": sha256_file(data_path)}
        },
        "limitations": [
            "评测规模为 30 个基础意图，优先保证成对结构和人工可核验性。",
            "字符串 pattern 是单一未见约束族，不代表所有现实能力表达。",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
