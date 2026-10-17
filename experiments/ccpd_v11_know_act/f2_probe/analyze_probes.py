#!/usr/bin/env python3
"""训练冻结线性探针并与浅层文本、合同特征基线比较。"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


def parse_args() -> argparse.Namespace:
    """解析冻结配对、隐藏状态与输出参数。"""
    parser = argparse.ArgumentParser(description="分析 CCPD-v1.1 F2 原生隐藏状态")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--hidden-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
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


def percentile(values: list[float], fraction: float) -> float:
    """使用线性插值计算百分位数。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_pairs(
    groups: list[dict[str, float]], statistic: Callable[[list[dict[str, float]]], float],
    repeats: int, seed: int,
) -> list[float]:
    """以基础请求配对为单位计算 Bootstrap 置信区间。"""
    rng = random.Random(seed)
    estimates = [
        statistic([groups[rng.randrange(len(groups))] for _ in groups])
        for _ in range(repeats)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """计算等宽分箱的 Expected Calibration Error。"""
    total = len(labels)
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not mask.any():
            continue
        accuracy = labels[mask].mean()
        confidence = probabilities[mask].mean()
        result += float(mask.sum()) / total * abs(float(accuracy - confidence))
    return result


def paired_metrics(
    rows: list[dict[str, Any]], probabilities: np.ndarray, repeats: int, seed: int
) -> dict[str, Any]:
    """计算分类、校准与成对排序指标。"""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row, score in zip(rows, probabilities):
        grouped[row["pair_id"]][row["side"]] = float(score)
    pairs = list(grouped.values())
    if any(set(pair) != {"plus", "minus"} for pair in pairs):
        raise RuntimeError("探针评测配对不完整")

    def concordance(sample: list[dict[str, float]]) -> float:
        """计算 E+ 分数高于 E- 的比例。"""
        return statistics.fmean(pair["plus"] > pair["minus"] for pair in sample)

    def score_difference(sample: list[dict[str, float]]) -> float:
        """计算成对概率差均值。"""
        return statistics.fmean(pair["plus"] - pair["minus"] for pair in sample)

    return {
        "sides": len(rows),
        "pairs": len(pairs),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece_10bin": expected_calibration_error(labels, probabilities),
        "paired_concordance": concordance(pairs),
        "paired_concordance_ci95": bootstrap_pairs(pairs, concordance, repeats, seed),
        "mean_pair_score_difference": score_difference(pairs),
        "mean_pair_score_difference_ci95": bootstrap_pairs(
            pairs, score_difference, repeats, seed + 1
        ),
        "pair_decision_success": statistics.fmean(
            pair["plus"] >= 0.5 and pair["minus"] < 0.5 for pair in pairs
        ),
        "mean_probability": float(probabilities.mean()),
    }


def fit_dense_probe(train_x: np.ndarray, train_y: np.ndarray, seed: int) -> Any:
    """拟合固定超参数的标准化逻辑回归。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    return model.fit(train_x, train_y)


def surface_features(pair: dict[str, Any], side_name: str) -> list[float]:
    """提取不读取用户请求的固定浅层合同特征。"""
    function = pair[side_name]["tools"][0]["function"]
    field = function["parameters"]["properties"][pair["target_parameter"]]
    serialized = json.dumps(function, ensure_ascii=False, sort_keys=True)
    key = next(name for name in ("enum", "minimum", "maximum", "const") if name in field)
    value = field[key][0] if key == "enum" else field[key]
    return [
        float(len(serialized)),
        float(len(re.findall(r"\S+", serialized))),
        float(len(function.get("description", "")) + len(field.get("description", ""))),
        float(len(function["parameters"]["properties"])),
        float(key == "enum"),
        float(key == "minimum"),
        float(key == "maximum"),
        float(key == "const"),
        float(len(str(value))),
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0,
    ]


def text_for(pair: dict[str, Any], side_name: str) -> str:
    """构造不含标签的用户请求与工具合同浅层文本。"""
    side = pair[side_name]
    return (
        side["messages"][0]["content"]
        + "\n"
        + json.dumps(side["tools"][0]["function"], ensure_ascii=False, sort_keys=True)
    )


def main() -> None:
    """比较固定层线性探针与两类浅层基线。"""
    args = parse_args()
    pairs = read_jsonl(args.pairs.resolve())
    pair_map = {row["pair_id"]: row for row in pairs}
    keys = read_jsonl(args.hidden_dir.resolve() / "keys.jsonl")
    arrays = np.load(args.hidden_dir.resolve() / "hidden_states.npz")
    if any(len(arrays[name]) != len(keys) for name in arrays.files):
        raise RuntimeError("隐藏状态与键数量不一致")
    train_indices = np.asarray([index for index, row in enumerate(keys) if row["split"] == "train"])
    train_y = np.asarray([keys[index]["label"] for index in train_indices], dtype=np.int64)
    eval_splits = ("dev", "test_id", "test_ood_family", "test_ood_constraint")
    result: dict[str, Any] = {
        "protocol": {
            "probe": "StandardScaler + LogisticRegression(C=1.0, class_weight=balanced)",
            "threshold": 0.5,
            "layer_selection": "预注册 embedding/middle/late/last；不使用测试集选层",
            "bootstrap_unit": "pair_id",
        },
        "representations": {},
    }

    for representation in arrays.files:
        model = fit_dense_probe(arrays[representation][train_indices], train_y, args.seed)
        result["representations"][representation] = {}
        for split in eval_splits:
            indices = np.asarray([index for index, row in enumerate(keys) if row["split"] == split])
            rows = [keys[index] for index in indices]
            probabilities = model.predict_proba(arrays[representation][indices])[:, 1]
            result["representations"][representation][split] = paired_metrics(
                rows, probabilities, args.bootstrap_repeats, args.seed
            )

    ordered_pairs = [pair_map[row["pair_id"]] for row in keys]
    surface_x = np.asarray(
        [surface_features(pair, row["side"]) for pair, row in zip(ordered_pairs, keys)],
        dtype=np.float64,
    )
    surface_model = fit_dense_probe(surface_x[train_indices], train_y, args.seed)
    result["representations"]["surface_contract"] = {}
    for split in eval_splits:
        indices = np.asarray([index for index, row in enumerate(keys) if row["split"] == split])
        probabilities = surface_model.predict_proba(surface_x[indices])[:, 1]
        result["representations"]["surface_contract"][split] = paired_metrics(
            [keys[index] for index in indices], probabilities, args.bootstrap_repeats, args.seed
        )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts = [text_for(pair, row["side"]) for pair, row in zip(ordered_pairs, keys)]
    text_model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=args.seed),
    )
    text_model.fit([texts[index] for index in train_indices], train_y)
    result["representations"]["char_tfidf"] = {}
    for split in eval_splits:
        indices = np.asarray([index for index, row in enumerate(keys) if row["split"] == split])
        probabilities = text_model.predict_proba([texts[index] for index in indices])[:, 1]
        result["representations"]["char_tfidf"][split] = paired_metrics(
            [keys[index] for index in indices], probabilities, args.bootstrap_repeats, args.seed
        )

    # 保存预注册 late 探针的逐样本概率，供 F3 使用；不根据测试结果改选层。
    late_model = fit_dense_probe(arrays["late"][train_indices], train_y, args.seed)
    late_probabilities = late_model.predict_proba(arrays["late"])[:, 1]
    result["late_predictions"] = [
        {
            **row,
            "feasible_probability": float(probability),
            "predicted_feasible": bool(probability >= 0.5),
        }
        for row, probability in zip(keys, late_probabilities)
    ]
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖 F2 分析：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    printable = {key: value for key, value in result.items() if key != "late_predictions"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
