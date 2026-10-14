#!/usr/bin/env python3
"""对冻结数据上的模型预测进行严格评测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评测工具调用预测")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--run-name", default="zero_shot")
    parser.add_argument("--report-title", default="Stage 1 Zero-shot Baseline 结果")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL 文件。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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


def percent(value: float | None) -> str:
    """将指标格式化为百分比。"""
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_markdown(summary: dict[str, Any], metadata: dict[str, Any], report_title: str) -> str:
    """渲染可直接进入实验报告的中文结果表。"""
    lines = [
        f"# {report_title}",
        "",
        "> 这是本项目冻结子集上的结果；BFCL-lite 不是 BFCL 官方排行榜成绩。",
        "",
        "## 运行信息",
        "",
        f"- 模型：`{metadata['model_path']}`",
        f"- 样本数：{metadata['samples']}",
        f"- 总耗时：{metadata['runtime_seconds']} 秒",
        f"- 峰值显存：{metadata['peak_memory_gib']} GiB",
        f"- 预测文件 SHA-256：`{metadata['prediction_sha256']}`",
        "",
        "## 分轨指标",
        "",
        "| 轨道 | 样本数 | 解析率 | Tool Accuracy | Arg Accuracy | Arg Exact | Execution Success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for track, metrics in summary["by_track"].items():
        lines.append(
            f"| {track} | {metrics['samples']} | {percent(metrics['parse_rate'])} | "
            f"{percent(metrics['tool_accuracy'])} | {percent(metrics['argument_accuracy'])} | "
            f"{percent(metrics['argument_exact_match'])} | {percent(metrics['execution_success_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 分类别指标",
            "",
            "| 类别 | 样本数 | Tool Accuracy | Arg Accuracy | Execution Success |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, metrics in summary["by_category"].items():
        lines.append(
            f"| {category} | {metrics['samples']} | {percent(metrics['tool_accuracy'])} | "
            f"{percent(metrics['argument_accuracy'])} | {percent(metrics['execution_success_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- Tool Accuracy：预测调用的工具名多重集合与标准答案完全一致。",
            "- Arg Accuracy：工具选择正确后按字段计算；工具选择错误时记为 0。",
            "- Arg Exact：所有参数字段完全正确，允许 BFCL 标注中声明的候选值。",
            "- Execution Success：仅在 controlled_v1 的可执行样本上统计，要求解析、工具、参数和 Schema 全部通过。",
            "- 评测器不修复非法 JSON，不对缺失参数进行猜测，也不静默回退。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """校验输入完整性、执行评测并保存明细和汇总。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.metrics import aggregate, evaluate_case

    frozen_dir = (
        args.frozen_dir.resolve()
        if args.frozen_dir is not None
        else project_root / "data" / "evaluation" / "frozen_v1"
    )
    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for filename, metadata in manifest["datasets"].items():
        path = frozen_dir / filename
        if sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"冻结数据哈希不匹配：{path}")
        samples.extend(read_jsonl(path))

    predictions = read_jsonl(args.predictions.resolve())
    prediction_map = {row["id"]: row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise RuntimeError("预测文件存在重复 ID")
    sample_ids = {row["id"] for row in samples}
    prediction_ids = set(prediction_map)
    if sample_ids != prediction_ids:
        missing = sorted(sample_ids - prediction_ids)
        extra = sorted(prediction_ids - sample_ids)
        raise RuntimeError(f"预测 ID 不完整，缺失={missing}，多余={extra}")

    detailed = [evaluate_case(sample, prediction_map[sample["id"]]) for sample in samples]
    summary = aggregate(detailed)
    metadata_path = args.predictions.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(args.predictions) != metadata["prediction_sha256"]:
        raise RuntimeError("预测文件哈希与运行元数据不一致")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.run_name.replace("_", "").isalnum():
        raise ValueError("run-name 只能包含字母、数字和下划线")
    detail_path = args.output_dir / f"{args.run_name}_case_results.jsonl"
    summary_path = args.output_dir / f"{args.run_name}_summary.json"
    report_path = args.output_dir / f"{args.run_name}_report.md"
    for path in (detail_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(f"评测输出已存在，拒绝覆盖：{path}")

    with detail_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in detailed:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(summary, metadata, args.report_title), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
