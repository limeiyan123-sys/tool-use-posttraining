#!/usr/bin/env python3
"""严格评测 Counterfactual Tool-World 预测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评测反事实工具世界预测")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
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


def percent(value: float) -> str:
    """将比例格式化为百分比。"""
    return f"{value * 100:.2f}%"


def render_report(
    summary: dict[str, Any], metadata: dict[str, Any], run_name: str, dataset_note: str
) -> str:
    """渲染中文工具世界诊断报告。"""
    lines = [
        f"# {run_name} 工具世界诊断结果",
        "",
        "## 运行信息",
        "",
        f"- 模型：`{metadata['model_path']}`",
        f"- 样本数：{metadata['samples']}",
        f"- 总耗时：{metadata['runtime_seconds']} 秒",
        f"- 峰值显存：{metadata['peak_memory_gib']} GiB",
        f"- 预测 SHA-256：`{metadata['prediction_sha256']}`",
        "",
        "## 分工具世界结果",
        "",
        "| Tool World | 样本数 | Tool Accuracy | Arg Accuracy | Arg Exact | Decision Exact | Semantic Exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for world_type, metrics in summary["by_world"].items():
        lines.append(
            f"| {world_type} | {metrics['samples']} | {percent(metrics['tool_accuracy'])} | "
            f"{percent(metrics['argument_accuracy'])} | {percent(metrics['argument_exact_match'])} | "
            f"{percent(metrics['decision_exact'])} | {percent(metrics['semantic_exact'])} |"
        )
    lines.extend(
        [
            "",
            "## 条件化指标",
            "",
            f"- Interface Equivariance Accuracy：{percent(summary['interface_equivariance_accuracy'])}",
            f"- All-worlds Group Accuracy：{percent(summary['all_worlds_group_accuracy']['accuracy'])}",
            "",
            "| Counterfactual Pair | Pair Accuracy | Both Correct / Pairs |",
            "|---|---:|---:|",
        ]
    )
    for world_type, metrics in summary["counterfactual_pair_accuracy"].items():
        lines.append(
            f"| original + {world_type} | {percent(metrics['accuracy'])} | "
            f"{metrics['both_correct']} / {metrics['pairs']} |"
        )
    lines.extend(
        [
            "",
            "## 评测纪律",
            "",
            "- 非法输出不自动修复。",
            "- 能力移除和替换后的 Gold 已按当前工具世界重新计算。",
            "- 多调用任务允许保留仍可独立完成的调用，不统一退化为 no-tool。",
            f"- {dataset_note}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """校验数据与预测完整性，计算并保存工具世界指标。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    frozen_dir = args.frozen_dir.resolve()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.tool_world_metrics import aggregate_tool_world, evaluate_tool_world_case

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
        raise RuntimeError(
            f"预测 ID 不完整，缺失={sorted(sample_ids - set(prediction_map))}，"
            f"多余={sorted(set(prediction_map) - sample_ids)}"
        )

    metadata_path = predictions_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(predictions_path) != metadata["prediction_sha256"]:
        raise RuntimeError("预测文件哈希与运行元数据不一致")
    detailed = [
        evaluate_tool_world_case(sample, prediction_map[sample["id"]]) for sample in samples
    ]
    summary = aggregate_tool_world(detailed)

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
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset_note = (
        "本数据为方法冻结后一次性使用的独立最终测试集，不再据此修改方法。"
        if manifest.get("version") == "tool_world_final_v1"
        else "本数据用于研究诊断与方法选择，不作为后续新方法的独立最终结论。"
    )
    report_path.write_text(
        render_report(summary, metadata, args.run_name, dataset_note), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
