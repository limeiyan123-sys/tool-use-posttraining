#!/usr/bin/env python3
"""审计 F0 表面捷径，并导出分层人工复核样本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析冻结数据与审计输出路径。"""
    parser = argparse.ArgumentParser(description="审计 CCPD-v1.1 F0 数据")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
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


def stable_rank(row: dict[str, Any], seed: int, label: str) -> str:
    """生成稳定抽查顺序。"""
    return hashlib.sha256(f"{seed}:{label}:{row['source_id']}".encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """创建 JSONL 审计产物并拒绝覆盖。"""
    if path.exists():
        raise FileExistsError(f"拒绝覆盖审计产物：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def choose_review(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """优先覆盖全部小划分，并从 OOD-family 与训练集补足 30 对。"""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    selected: list[dict[str, Any]] = []
    quotas = {
        "dev": len(by_split["dev"]),
        "test_id": len(by_split["test_id"]),
        "test_ood_constraint": len(by_split["test_ood_constraint"]),
        "test_ood_family": 8,
        "train": 4,
    }
    for split, quota in quotas.items():
        ordered = sorted(by_split[split], key=lambda row: stable_rank(row, seed, split))
        selected.extend(ordered[:quota])
    if len(selected) != 30:
        raise RuntimeError(f"人工复核样本数量异常：{len(selected)}/30")
    return selected


def surface_features(pair: dict[str, Any], side_name: str) -> list[float]:
    """提取不读取用户请求的浅层合同特征。"""
    side = pair[side_name]
    function = side["tools"][0]["function"]
    target = pair["target_parameter"]
    field = function["parameters"]["properties"][target]
    serialized = json.dumps(function, ensure_ascii=False, sort_keys=True)
    description = f"{function.get('description', '')} {field.get('description', '')}"
    constraint_key = next(
        key for key in ("enum", "minimum", "maximum", "const") if key in field
    )
    value = field[constraint_key]
    if constraint_key == "enum":
        scalar = value[0]
        cardinality = len(value)
    else:
        scalar = value
        cardinality = 1
    return [
        float(len(serialized)),
        float(len(re.findall(r"\S+", serialized))),
        float(len(description)),
        float(len(function["parameters"]["properties"])),
        float(cardinality),
        float(constraint_key == "enum"),
        float(constraint_key == "minimum"),
        float(constraint_key == "maximum"),
        float(constraint_key == "const"),
        float(len(str(scalar))),
        float(isinstance(scalar, (int, float)) and not isinstance(scalar, bool)),
        float(scalar) if isinstance(scalar, (int, float)) and not isinstance(scalar, bool) else 0.0,
    ]


def build_xy(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[int], list[tuple[str, str]]]:
    """展开配对两侧为浅层分类输入。"""
    features: list[list[float]] = []
    labels: list[int] = []
    keys: list[tuple[str, str]] = []
    for row in rows:
        for side, label in (("plus", 1), ("minus", 0)):
            features.append(surface_features(row, side))
            labels.append(label)
            keys.append((row["pair_id"], side))
    return features, labels, keys


def evaluate_surface_model(
    train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    """使用固定逻辑回归检查长度、键名和约束值等浅层线索。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x, train_y, _ = build_xy(train_rows)
    test_x, test_y, keys = build_xy(test_rows)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed),
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(test_x)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for key, score in zip(keys, scores):
        by_pair[key[0]][key[1]] = float(score)
    concordance = sum(values["plus"] > values["minus"] for values in by_pair.values()) / len(by_pair)
    return {
        "sides": len(test_y),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "auroc": float(roc_auc_score(test_y, scores)),
        "paired_concordance": concordance,
    }


def main() -> None:
    """运行固定浅层基线并导出人工复核样本。"""
    args = parse_args()
    rows = read_jsonl(args.pairs.resolve())
    train_rows = [row for row in rows if row["split"] == "train"]
    result = {
        "feature_policy": "不读取用户请求，仅使用序列化长度、Token 数、字段数、约束键、基数和约束标量",
        "by_split": {},
    }
    for split in ("dev", "test_id", "test_ood_family", "test_ood_constraint"):
        split_rows = [row for row in rows if row["split"] == split]
        result["by_split"][split] = evaluate_surface_model(train_rows, split_rows, args.seed)
    review = choose_review(rows, args.seed)
    write_jsonl(args.review_output.resolve(), review)
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖 F0 审计：{args.output}")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
