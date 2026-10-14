#!/usr/bin/env python3
"""对两个工具调用方法执行逐样本配对比较。"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable


METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "tool_accuracy": lambda row: float(row["tool_correct"]),
    "argument_accuracy": lambda row: float(row["argument_accuracy"]),
    "argument_exact_match": lambda row: float(row["argument_exact"]),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="配对比较两组工具调用结果")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261014)
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """读取结果明细，并拒绝重复样本 ID。"""
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            sample_id = row["id"]
            if sample_id in rows:
                raise ValueError(f"{path}:{line_number} 出现重复 ID：{sample_id}")
            rows[sample_id] = row
    return rows


def paired_bootstrap(
    baseline: list[float], candidate: list[float], count: int, rng: random.Random
) -> dict[str, float]:
    """计算候选方法相对基线的配对 bootstrap 差值区间。"""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("配对 bootstrap 输入长度非法")
    size = len(baseline)
    observed = sum(c - b for b, c in zip(baseline, candidate)) / size
    estimates: list[float] = []
    for _ in range(count):
        indices = [rng.randrange(size) for _ in range(size)]
        estimates.append(sum(candidate[i] - baseline[i] for i in indices) / size)
    estimates.sort()
    lower = estimates[int(0.025 * count)]
    upper = estimates[min(count - 1, int(0.975 * count))]
    return {"delta": observed, "ci95_low": lower, "ci95_high": upper}


def summarize_group(
    ids: list[str], baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]],
    count: int, rng: random.Random
) -> dict[str, Any]:
    """汇总一个样本组的指标及其配对差异。"""
    result: dict[str, Any] = {"samples": len(ids), "metrics": {}}
    for name, extractor in METRICS.items():
        baseline_values = [extractor(baseline[sample_id]) for sample_id in ids]
        candidate_values = [extractor(candidate[sample_id]) for sample_id in ids]
        result["metrics"][name] = {
            "baseline": sum(baseline_values) / len(ids),
            "candidate": sum(candidate_values) / len(ids),
            **paired_bootstrap(baseline_values, candidate_values, count, rng),
        }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    """渲染中文配对比较报告。"""
    lines = [
        "# Stage 1 Zero-shot 与 LoRA SFT 配对比较",
        "",
        f"> 对同一冻结测试集逐样本比较；95% CI 由 {result['bootstrap_samples']} 次配对 bootstrap 得到。",
        "",
        "## 总体指标",
        "",
        "| 指标 | Zero-shot | LoRA SFT | 差值 | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in result["overall"]["metrics"].items():
        lines.append(
            f"| {name} | {values['baseline']:.2%} | {values['candidate']:.2%} | "
            f"{values['delta']:+.2%} | [{values['ci95_low']:+.2%}, {values['ci95_high']:+.2%}] |"
        )
    lines.extend(["", "## 分轨比较", "", "| 轨道 | 样本数 | Tool 差值 | Arg 差值 | Arg Exact 差值 |", "|---|---:|---:|---:|---:|"])
    for track, group in result["by_track"].items():
        metrics = group["metrics"]
        lines.append(
            f"| {track} | {group['samples']} | {metrics['tool_accuracy']['delta']:+.2%} | "
            f"{metrics['argument_accuracy']['delta']:+.2%} | "
            f"{metrics['argument_exact_match']['delta']:+.2%} |"
        )
    lines.extend(["", "## 错误迁移", ""])
    for transition, count in result["error_transitions"].items():
        lines.append(f"- `{transition}`：{count}")
    lines.extend(["", "## 解释边界", "", "- 置信区间反映当前 98 条冻结样本上的抽样不确定性，不代表 BFCL 官方总体。", "- 若区间跨过 0，只能报告观察到的趋势，不能宣称统计显著改进。", ""])
    return "\n".join(lines)


def main() -> None:
    """校验样本对应关系并生成 JSON 与 Markdown 比较结果。"""
    args = parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("bootstrap-samples 不得小于 1000")
    for path in (args.output_json, args.output_md):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    baseline = read_jsonl(args.baseline)
    candidate = read_jsonl(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("两组结果的样本 ID 不一致")

    ids = sorted(baseline)
    rng = random.Random(args.seed)
    tracks = sorted({baseline[sample_id]["track"] for sample_id in ids})
    transitions = Counter(
        f"{baseline[sample_id]['error_type']} -> {candidate[sample_id]['error_type']}"
        for sample_id in ids
    )
    result = {
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "overall": summarize_group(ids, baseline, candidate, args.bootstrap_samples, rng),
        "by_track": {
            track: summarize_group(
                [sample_id for sample_id in ids if baseline[sample_id]["track"] == track],
                baseline,
                candidate,
                args.bootstrap_samples,
                rng,
            )
            for track in tracks
        },
        "error_transitions": dict(sorted(transitions.items())),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
