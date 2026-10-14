#!/usr/bin/env python3
"""严格评测 FCTW 四象限预测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析评测输入与输出参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取非空 JSONL 对象。"""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number} 出现空行")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} 不是 JSON 对象")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: float) -> str:
    """格式化百分比。"""
    return f"{value * 100:.2f}%"


def render_report(summary: dict[str, Any], metadata: dict[str, Any], run_name: str) -> str:
    """生成简体中文评测报告。"""
    lines = [
        f"# {run_name} FCTW 独立评测",
        "",
        "## 运行信息",
        "",
        f"- 模型：`{metadata['model_path']}`",
        f"- 样本数：{metadata['samples']}",
        f"- 总耗时：{metadata['runtime_seconds']} 秒",
        f"- 峰值显存：{metadata['peak_memory_gib']} GiB",
        f"- 预测 SHA-256：`{metadata['prediction_sha256']}`",
        "",
        "## 四象限",
        "",
        "| 世界 | 样本数 | Decision Exact | Tool Accuracy | Arg Exact | False Rejection | Wrong Tool |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for world, metrics in summary["by_world"].items():
        lines.append(
            f"| {world} | {metrics['samples']} | {percent(metrics['decision_exact'])} | "
            f"{percent(metrics['tool_accuracy'])} | {percent(metrics['argument_exact_match'])} | "
            f"{percent(metrics['false_rejection_rate'])} | "
            f"{percent(metrics['wrong_tool_selection_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 约束泛化",
            "",
            "| 约束族 | 样本数 | Decision Exact | False Rejection | Wrong Tool |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, metrics in summary["by_constraint"].items():
        suffix = "（训练未见）" if family == "string_pattern" else ""
        lines.append(
            f"| {family}{suffix} | {metrics['samples']} | {percent(metrics['decision_exact'])} | "
            f"{percent(metrics['false_rejection_rate'])} | "
            f"{percent(metrics['wrong_tool_selection_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 关键象限分约束结果",
            "",
            "| 约束族 | E01 Decision | E11 Decision | E11 False Rejection |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for family, worlds in summary["by_constraint_world"].items():
        lines.append(
            f"| {family} | {percent(worlds['E01']['decision_exact'])} | "
            f"{percent(worlds['E11']['decision_exact'])} | "
            f"{percent(worlds['E11']['false_rejection_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"- Overall Decision Exact：{percent(summary['overall']['decision_exact'])}",
            f"- E01 Capability Substitution Accuracy：{percent(summary['capability_substitution_accuracy'])}",
            f"- All-worlds Group Accuracy：{percent(summary['all_worlds_group_accuracy']['accuracy'])}",
            "",
            "## 评测纪律",
            "",
            "- 30 个基础意图均未用于 FCTW 训练或此前诊断与最终评测。",
            "- 同一意图的四个世界按组统计，工具数量保持一致。",
            "- `string_pattern` 为训练阶段未出现的能力约束族。",
            "- 非法输出不自动修复。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """校验冻结数据与预测哈希，计算并保存完整评测产物。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    frozen_dir = args.frozen_dir.resolve()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.fctw_metrics import aggregate_fctw, evaluate_fctw_case

    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for filename, file_metadata in manifest["datasets"].items():
        path = frozen_dir / filename
        if sha256_file(path) != file_metadata["sha256"]:
            raise RuntimeError(f"冻结数据哈希不匹配：{path}")
        samples.extend(read_jsonl(path))
    predictions = read_jsonl(predictions_path)
    prediction_map = {row["id"]: row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise RuntimeError("预测文件存在重复 ID")
    sample_ids = {row["id"] for row in samples}
    if sample_ids != set(prediction_map):
        raise RuntimeError("预测 ID 与 FCTW 冻结评测集不一致")
    metadata = json.loads(
        predictions_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    if sha256_file(predictions_path) != metadata["prediction_sha256"]:
        raise RuntimeError("预测文件哈希与元数据不一致")
    detailed = [
        evaluate_fctw_case(sample, prediction_map[sample["id"]]) for sample in samples
    ]
    summary = aggregate_fctw(detailed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.run_name.replace("_", "").isalnum():
        raise ValueError("run-name 只能包含字母、数字和下划线")
    detail_path = output_dir / f"{args.run_name}_case_results.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    report_path = output_dir / f"{args.run_name}_report.md"
    for path in (detail_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(f"评测输出已存在，拒绝覆盖：{path}")
    with detail_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in detailed:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_report(summary, metadata, args.run_name), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
