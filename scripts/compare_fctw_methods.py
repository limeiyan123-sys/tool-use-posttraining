#!/usr/bin/env python3
"""比较 FDSRC、FWCT 与 FCTW 的四象限结果并执行按意图聚类的 Bootstrap。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    """解析三组逐样本结果和输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdsrc", type=Path, required=True)
    parser.add_argument("--fwct", type=Path, required=True)
    parser.add_argument("--fctw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取逐样本 JSONL 并拒绝重复 ID。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"结果存在重复 ID：{path}")
    return rows


def percentile(values: list[float], probability: float) -> float:
    """计算经验分位数。"""
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def clustered_bootstrap(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    selector: Callable[[dict[str, Any]], bool],
    value: Callable[[dict[str, Any]], float],
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """以 base_id 为抽样单位估计候选减参考的置信区间。"""
    selected_ids = [row_id for row_id, row in reference.items() if selector(row)]
    if not selected_ids:
        raise ValueError("统计范围内没有样本")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row_id in selected_ids:
        if row_id not in candidate:
            raise ValueError(f"候选结果缺少 ID：{row_id}")
        grouped[reference[row_id]["base_id"]].append(
            value(candidate[row_id]) - value(reference[row_id])
        )
    cluster_values = {
        base_id: sum(values) / len(values) for base_id, values in grouped.items()
    }
    cluster_ids = sorted(cluster_values)
    rng = random.Random(seed)
    distribution = [
        sum(cluster_values[rng.choice(cluster_ids)] for _ in cluster_ids) / len(cluster_ids)
        for _ in range(samples)
    ]
    observed = sum(cluster_values.values()) / len(cluster_values)
    return {
        "clusters": len(cluster_ids),
        "difference": observed,
        "ci_low": percentile(distribution, 0.025),
        "ci_high": percentile(distribution, 0.975),
    }


def method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """提取比较表所需核心指标。"""
    def rate(selected: list[dict[str, Any]], key: str) -> float:
        return sum(float(row[key]) for row in selected) / len(selected)

    by_world = {
        world: [row for row in rows if row["world_type"] == world]
        for world in ("E10", "E01", "E11", "E00")
    }
    unseen = [row for row in rows if row["constraint_family"] == "string_pattern"]
    return {
        "samples": len(rows),
        "overall_decision_exact": rate(rows, "decision_exact"),
        "E10_decision_exact": rate(by_world["E10"], "decision_exact"),
        "E01_decision_exact": rate(by_world["E01"], "decision_exact"),
        "E11_decision_exact": rate(by_world["E11"], "decision_exact"),
        "E00_decision_exact": rate(by_world["E00"], "decision_exact"),
        "E11_false_rejection_rate": sum(
            row["fctw_error"] == "false_rejection" for row in by_world["E11"]
        )
        / len(by_world["E11"]),
        "unseen_constraint_decision_exact": rate(unseen, "decision_exact"),
    }


def percent(value: float) -> str:
    """格式化百分比。"""
    return f"{value * 100:.2f}%"


def render_report(result: dict[str, Any]) -> str:
    """生成简体中文方法比较报告。"""
    lines = [
        "# FCTW 核心方法比较",
        "",
        "| 方法 | Overall | E10 | E01 | E11 | E00 | E11 False Rejection | Unseen Constraint |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("fdsrc", "fwct", "fctw"):
        metrics = result["methods"][method]
        lines.append(
            f"| {method.upper()} | {percent(metrics['overall_decision_exact'])} | "
            f"{percent(metrics['E10_decision_exact'])} | {percent(metrics['E01_decision_exact'])} | "
            f"{percent(metrics['E11_decision_exact'])} | {percent(metrics['E00_decision_exact'])} | "
            f"{percent(metrics['E11_false_rejection_rate'])} | "
            f"{percent(metrics['unseen_constraint_decision_exact'])} |"
        )
    lines.extend(["", "## 按意图聚类的配对差值", ""])
    for comparison, scopes in result["comparisons"].items():
        lines.append(f"### {comparison}")
        lines.append("")
        lines.append("| 指标 | 差值 | 95% CI | 意图组 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, stats in scopes.items():
            lines.append(
                f"| {name} | {stats['difference'] * 100:+.2f} pp | "
                f"[{stats['ci_low'] * 100:+.2f}, {stats['ci_high'] * 100:+.2f}] pp | "
                f"{stats['clusters']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """完成一致性校验、聚类统计并写出结果。"""
    args = parse_args()
    rows_by_method = {
        "fdsrc": read_jsonl(args.fdsrc.resolve()),
        "fwct": read_jsonl(args.fwct.resolve()),
        "fctw": read_jsonl(args.fctw.resolve()),
    }
    indexes = {
        method: {row["id"]: row for row in rows}
        for method, rows in rows_by_method.items()
    }
    id_sets = {method: set(index) for method, index in indexes.items()}
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise RuntimeError("三种方法的评测 ID 不一致")
    result: dict[str, Any] = {
        "methods": {
            method: method_summary(rows) for method, rows in rows_by_method.items()
        },
        "comparisons": {},
    }
    scopes: dict[str, tuple[Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], float]]] = {
        "Overall Decision Exact": (lambda row: True, lambda row: float(row["decision_exact"])),
        "E01 Decision Exact": (
            lambda row: row["world_type"] == "E01",
            lambda row: float(row["decision_exact"]),
        ),
        "E11 Decision Exact": (
            lambda row: row["world_type"] == "E11",
            lambda row: float(row["decision_exact"]),
        ),
        "E11 False Rejection": (
            lambda row: row["world_type"] == "E11",
            lambda row: float(row["fctw_error"] == "false_rejection"),
        ),
        "Unseen Constraint Decision Exact": (
            lambda row: row["constraint_family"] == "string_pattern",
            lambda row: float(row["decision_exact"]),
        ),
    }
    for reference_name in ("fdsrc", "fwct"):
        comparison = f"FCTW - {reference_name.upper()}"
        result["comparisons"][comparison] = {
            name: clustered_bootstrap(
                indexes[reference_name],
                indexes["fctw"],
                selector,
                value,
                args.bootstrap_samples,
                args.seed + index,
            )
            for index, (name, (selector, value)) in enumerate(scopes.items())
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "fctw_comparison.json"
    report_path = args.output_dir / "fctw_comparison.md"
    for path in (json_path, report_path):
        if path.exists():
            raise FileExistsError(f"比较输出已存在，拒绝覆盖：{path}")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
