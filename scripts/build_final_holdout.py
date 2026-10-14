#!/usr/bin/env python3
"""从未使用的 BFCL 样本中冻结独立最终评测集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_frozen_eval import (
    BFCL_CATEGORIES,
    convert_ground_truth,
    git_commit,
    normalize_schema,
    read_jsonl,
    sha256_file,
    stable_select,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建独立 BFCL 最终 holdout")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    return parser.parse_args()


def used_bfcl_ids(frozen_dir: Path) -> set[str]:
    """读取开发诊断集已使用的 BFCL 原始 ID。"""
    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    used: set[str] = set()
    for filename in manifest["datasets"]:
        for row in read_jsonl(frozen_dir / filename):
            source_id = row.get("source_id")
            if isinstance(source_id, str):
                used.add(source_id)
    return used


def build_holdout(project_root: Path, seed: int, used_ids: set[str]) -> list[dict[str, Any]]:
    """按稳定哈希选择与开发集互斥的 BFCL 样本。"""
    data_root = (
        project_root
        / "vendor"
        / "gorilla"
        / "berkeley-function-call-leaderboard"
        / "bfcl_eval"
        / "data"
    )
    answer_root = data_root / "possible_answer"
    output: list[dict[str, Any]] = []
    for category, size in BFCL_CATEGORIES.items():
        source_path = data_root / f"BFCL_v4_{category}.json"
        candidates = [row for row in read_jsonl(source_path) if row["id"] not in used_ids]
        selected = stable_select(candidates, size=size, seed=seed)
        answers: dict[str, list[dict[str, Any]]] = {}
        if category != "irrelevance":
            answer_path = answer_root / f"BFCL_v4_{category}.json"
            answers = {row["id"]: row["ground_truth"] for row in read_jsonl(answer_path)}
        for row in selected:
            turns = row["question"]
            if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], list):
                raise ValueError(f"{row['id']} 不是受支持的单轮 BFCL 格式")
            expected = [] if category == "irrelevance" else convert_ground_truth(answers[row["id"]])
            output.append(
                {
                    "id": f"holdout_{row['id']}",
                    "source_id": row["id"],
                    "track": "bfcl_holdout_v1",
                    "category": category,
                    "messages": turns[0],
                    "tools": [
                        {"type": "function", "function": normalize_schema(function)}
                        for function in row["function"]
                    ],
                    "expected_calls": expected,
                    "executable": False,
                }
            )
    return sorted(output, key=lambda row: row["id"])


def main() -> None:
    """创建不可覆盖的最终 holdout 及其完整性清单。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    development_dir = project_root / "data" / "evaluation" / "frozen_v1"
    output_dir = project_root / "data" / "evaluation" / "final_holdout_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "bfcl_holdout_v1.jsonl"
    manifest_path = output_dir / "manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"最终 holdout 已存在，拒绝覆盖：{output_dir}")

    used_ids = used_bfcl_ids(development_dir)
    rows = build_holdout(project_root, args.seed, used_ids)
    if any(row["source_id"] in used_ids for row in rows):
        raise RuntimeError("最终 holdout 与开发诊断集发生重叠")
    write_jsonl(data_path, rows)
    gorilla_root = project_root / "vendor" / "gorilla"
    manifest = {
        "version": "final_holdout_v1",
        "seed": args.seed,
        "policy": "方法定型后创建；所有模型仅运行一次；文件禁止覆盖。",
        "development_set": str(development_dir),
        "excluded_source_ids": len(used_ids),
        "gorilla_commit": git_commit(gorilla_root),
        "datasets": {
            data_path.name: {"samples": len(rows), "sha256": sha256_file(data_path)}
        },
        "bfcl_category_sizes": BFCL_CATEGORIES,
        "limitations": [
            "这是固定 BFCL 子集，不是 BFCL 官方排行榜成绩。",
            "公开 BFCL 样本可能进入过基础模型预训练语料。",
            "holdout 不含本地可执行轨道，Execution Success 不适用。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
