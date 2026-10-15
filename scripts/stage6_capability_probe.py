#!/usr/bin/env python3
"""构建、运行并评测 Stage 6 单工具能力验证 Probe。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析三个子命令的参数。"""
    parser = argparse.ArgumentParser(description="Stage 6 单工具能力验证 Probe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="从 FCTW 独立评测集构造 Probe")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="运行原协议与显式兼容性判断")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--adapter-path", type=Path)
    run.add_argument("--output-path", type=Path, required=True)
    run.add_argument("--seed", type=int, default=20261022)
    run.add_argument("--direct-max-new-tokens", type=int, default=128)
    run.add_argument("--explicit-max-new-tokens", type=int, default=64)
    run.add_argument("--resume-partial", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="计算 Probe 指标")
    evaluate.add_argument("--project-root", type=Path, required=True)
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--run-name", required=True)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以稳定键顺序创建 JSONL，拒绝覆盖已有结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_tool(row: dict[str, Any], name: str) -> dict[str, Any]:
    """按名称取得唯一工具，缺失或重复时抛出异常。"""
    matches = [tool for tool in row["tools"] if tool["function"]["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"{row['id']} 中工具 {name} 的数量为 {len(matches)}")
    return matches[0]


def user_text(row: dict[str, Any]) -> str:
    """提取单轮用户请求，拒绝含糊的多消息格式。"""
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(f"{row['id']} 不是受支持的单轮用户请求")
    content = messages[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{row['id']} 缺少用户请求文本")
    return content


def expected_violation(row: dict[str, Any]) -> str:
    """从确定性变换记录生成字段级约束定位标签。"""
    transformation = row["transformation"]
    target = transformation["target_parameter"]
    constraint = transformation["constraint"]
    if not isinstance(constraint, dict) or len(constraint) != 1:
        raise ValueError(f"{row['id']} 的约束必须只有一个可验证键")
    constraint_name = next(iter(constraint))
    return f"{target}.{constraint_name}"


def make_probe_sample(
    row: dict[str, Any], tool: dict[str, Any], sample_type: str, compatible: bool,
    violation: str | None,
) -> dict[str, Any]:
    """构造单工具 Probe 样本。"""
    return {
        "id": f"{row['base_id']}__{sample_type}",
        "base_id": row["base_id"],
        "source_id": row["source_id"],
        "category": row["category"],
        "constraint_family": (
            "functional_mismatch"
            if sample_type == "functional_mismatch"
            else row["transformation"]["constraint_family"]
        ),
        "sample_type": sample_type,
        "messages": [{"role": "user", "content": user_text(row)}],
        "tools": [tool],
        "expected_compatible": compatible,
        "expected_violation": violation,
    }


def build_probe(args: argparse.Namespace) -> None:
    """从独立 FCTW 四象限评测集构造 30 组、90 条单工具样本。"""
    project_root = args.project_root.resolve()
    source_dir = project_root / "data" / "evaluation" / "fctw_eval_v1"
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    source_path = source_dir / "fctw_eval_v1.jsonl"
    expected_hash = source_manifest["datasets"][source_path.name]["sha256"]
    if sha256_file(source_path) != expected_hash:
        raise RuntimeError("FCTW 独立评测集哈希不匹配")

    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(source_path):
        groups[row["base_id"]][row["world_type"]] = row
    rows: list[dict[str, Any]] = []
    for base_id in sorted(groups):
        worlds = groups[base_id]
        if set(worlds) != {"E00", "E01", "E10", "E11"}:
            raise RuntimeError(f"{base_id} 的四象限不完整：{sorted(worlds)}")
        negative = worlds["E01"]
        positive = worlds["E10"]
        unrelated = worlds["E00"]
        near_miss_name = negative["transformation"]["near_miss_tool"]
        valid_name = positive["transformation"]["valid_tool"]
        rows.append(
            make_probe_sample(
                negative,
                find_tool(negative, near_miss_name),
                "near_miss",
                False,
                expected_violation(negative),
            )
        )
        rows.append(
            make_probe_sample(
                positive, find_tool(positive, valid_name), "valid", True, None
            )
        )
        unrelated_tool = sorted(unrelated["tools"], key=lambda item: item["function"]["name"])[0]
        rows.append(
            make_probe_sample(
                unrelated, unrelated_tool, "functional_mismatch", False, "tool.function"
            )
        )

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Probe 样本 ID 重复")
    output_dir = args.output_dir.resolve()
    data_path = output_dir / "capability_probe_v1.jsonl"
    manifest_path = output_dir / "manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Probe 输出已存在，拒绝覆盖：{output_dir}")
    write_jsonl(data_path, rows)
    sample_type_counts: dict[str, int] = defaultdict(int)
    constraint_family_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        sample_type_counts[row["sample_type"]] += 1
        constraint_family_counts[row["constraint_family"]] += 1
    manifest = {
        "version": "capability_probe_v1",
        "seed": source_manifest["seed"],
        "source": str(source_path.relative_to(project_root)),
        "source_sha256": expected_hash,
        "base_tasks": len(groups),
        "samples": len(rows),
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "constraint_family_counts": dict(sorted(constraint_family_counts.items())),
        "protocols": ["direct_tool_call", "explicit_compatibility_json"],
        "data_sha256": sha256_file(data_path),
        "limitations": [
            "能力约束覆盖 enum、numeric_range、string_pattern 与功能不匹配。",
            "未覆盖需要澄清、多步执行和外部权限等非 Schema 能力。",
            "显式协议未经过训练，其格式错误与语义错误必须分开报告。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_probe(dataset_dir: Path) -> list[dict[str, Any]]:
    """读取并校验冻结 Probe 数据。"""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    data_path = dataset_dir / "capability_probe_v1.jsonl"
    if sha256_file(data_path) != manifest["data_sha256"]:
        raise RuntimeError("Probe 数据哈希不匹配")
    rows = read_jsonl(data_path)
    if len(rows) != manifest["samples"]:
        raise RuntimeError("Probe 样本数与清单不一致")
    return rows


def direct_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """构造与历史工具调用评测一致的单工具消息。"""
    system = {
        "role": "system",
        "content": (
            "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
            "否则不要调用工具。需要调用时只输出结构化工具调用。"
        ),
    }
    return [system, *sample["messages"]]


def explicit_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """构造固定格式的显式兼容性判断消息。"""
    tool_json = json.dumps(sample["tools"][0]["function"], ensure_ascii=False, sort_keys=True)
    query = sample["messages"][0]["content"]
    system = (
        "你是工具能力验证器。判断唯一候选工具能否完整满足用户请求。"
        "只输出一个 JSON 对象，必须且只能包含 compatible 和 violated_constraint。"
        "兼容时输出 {\"compatible\":true,\"violated_constraint\":null}；"
        "不兼容时输出 {\"compatible\":false,\"violated_constraint\":\"字段.约束\"}。"
        "功能不匹配使用 tool.function。不要调用工具，不要解释。"
    )
    user = f"用户请求：\n{query}\n\n候选工具：\n{tool_json}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_one(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    """执行一次确定性生成并返回响应、Token 数和延迟。"""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return response, int(inputs["input_ids"].shape[1]), int(new_tokens.shape[0]), time.perf_counter() - started


def run_probe(args: argparse.Namespace) -> None:
    """加载一个模型并依次运行两种互补 Probe 协议。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project_root = args.project_root.resolve()
    dataset_dir = args.dataset_dir.resolve()
    model_path = args.model_path.resolve()
    adapter_path = args.adapter_path.resolve() if args.adapter_path else None
    output_path = args.output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"预测文件已存在，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_probe(dataset_dir)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=False,
    )
    model = (
        PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
        if adapter_path
        else base_model
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    existing: dict[str, dict[str, Any]] = {}
    mode = "x"
    if partial_path.exists():
        if not args.resume_partial:
            raise FileExistsError(f"发现临时文件，需显式使用 --resume-partial：{partial_path}")
        partial_rows = read_jsonl(partial_path)
        existing = {row["id"]: row for row in partial_rows}
        if len(existing) != len(partial_rows):
            raise RuntimeError("临时预测中存在重复 ID")
        mode = "a"
        print(f"[WARNING] 续跑已校验的 {len(existing)} 条 Probe 预测")

    started = time.perf_counter()
    with partial_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(rows, start=1):
            if sample["id"] in existing:
                continue
            direct_prompt = tokenizer.apply_chat_template(
                direct_messages(sample), tools=sample["tools"], tokenize=False, add_generation_prompt=True
            )
            explicit_prompt = tokenizer.apply_chat_template(
                explicit_messages(sample), tokenize=False, add_generation_prompt=True
            )
            direct = generate_one(model, tokenizer, direct_prompt, args.direct_max_new_tokens)
            explicit = generate_one(model, tokenizer, explicit_prompt, args.explicit_max_new_tokens)
            record = {
                "id": sample["id"],
                "direct_response": direct[0],
                "explicit_response": explicit[0],
                "direct_input_tokens": direct[1],
                "direct_output_tokens": direct[2],
                "direct_latency_seconds": round(direct[3], 6),
                "explicit_input_tokens": explicit[1],
                "explicit_output_tokens": explicit[2],
                "explicit_latency_seconds": round(explicit[3], 6),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(f"已完成 {index}/{len(rows)}")
    partial_path.replace(output_path)
    metadata = {
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "seed": args.seed,
        "samples": len(rows),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "prediction_sha256": sha256_file(output_path),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_explicit(text: str) -> tuple[bool, bool | None, str | None, str | None]:
    """严格解析显式兼容性 JSON，不自动提取或修复。"""
    cleaned = text.replace("<|im_end|>", "").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        return False, None, None, error.msg
    if not isinstance(value, dict) or set(value) != {"compatible", "violated_constraint"}:
        return False, None, None, "必须且只能包含 compatible、violated_constraint"
    if not isinstance(value["compatible"], bool):
        return False, None, None, "compatible 不是布尔值"
    violation = value["violated_constraint"]
    if violation is not None and not isinstance(violation, str):
        return False, None, None, "violated_constraint 不是字符串或 null"
    return True, value["compatible"], violation, None


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合兼容性、格式、定位和跨协议一致性指标。"""
    total = len(cases)
    direct_valid = sum(case["direct_format_valid"] for case in cases)
    explicit_valid = sum(case["explicit_format_valid"] for case in cases)
    direct_correct = sum(case["direct_compatibility_correct"] for case in cases)
    explicit_correct = sum(case["explicit_compatibility_correct"] for case in cases)
    semantic_rows = [case for case in cases if case["explicit_format_valid"]]
    incompatible_rows = [case for case in cases if not case["expected_compatible"]]
    localization = sum(case["violation_localization_correct"] for case in incompatible_rows)
    field_localization = sum(case["violation_field_correct"] for case in incompatible_rows)
    constraint_localization = sum(
        case["violation_constraint_correct"] for case in incompatible_rows
    )
    consistency_rows = [
        case for case in cases if case["direct_format_valid"] and case["explicit_format_valid"]
    ]
    consistent = sum(case["decision_consistent"] for case in consistency_rows)
    return {
        "samples": total,
        "direct_format_rate": direct_valid / total if total else None,
        "direct_compatibility_accuracy": direct_correct / total if total else None,
        "explicit_format_rate": explicit_valid / total if total else None,
        "explicit_compatibility_accuracy": explicit_correct / total if total else None,
        "explicit_semantic_accuracy_when_parseable": (
            sum(case["explicit_compatibility_correct"] for case in semantic_rows) / len(semantic_rows)
            if semantic_rows else None
        ),
        "violation_localization_accuracy": localization / len(incompatible_rows) if incompatible_rows else None,
        "violation_field_accuracy": field_localization / len(incompatible_rows) if incompatible_rows else None,
        "violation_constraint_accuracy": (
            constraint_localization / len(incompatible_rows) if incompatible_rows else None
        ),
        "decision_consistency": consistent / len(consistency_rows) if consistency_rows else None,
    }


def percent(value: float | None) -> str:
    """格式化可空百分比。"""
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_probe_report(run_name: str, summary: dict[str, Any]) -> str:
    """渲染单模型 Probe 报告。"""
    overall = summary["overall"]
    lines = [
        f"# {run_name} 单工具能力验证 Probe",
        "",
        "## 总体结果",
        "",
        f"- 样本数：{overall['samples']}",
        f"- 原协议格式有效率：{percent(overall['direct_format_rate'])}",
        f"- 原协议 Compatibility Accuracy：{percent(overall['direct_compatibility_accuracy'])}",
        f"- 显式协议格式有效率：{percent(overall['explicit_format_rate'])}",
        f"- 显式 Compatibility Accuracy：{percent(overall['explicit_compatibility_accuracy'])}",
        f"- 显式可解析样本语义准确率：{percent(overall['explicit_semantic_accuracy_when_parseable'])}",
        f"- Violation Localization Accuracy：{percent(overall['violation_localization_accuracy'])}",
        f"- Violation Field Accuracy：{percent(overall['violation_field_accuracy'])}",
        f"- Violation Constraint Accuracy：{percent(overall['violation_constraint_accuracy'])}",
        f"- Decision Consistency：{percent(overall['decision_consistency'])}",
        "",
        "## 分样本类型",
        "",
        "| 类型 | N | Direct Compat | Explicit Compat | Localization | Consistency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["by_sample_type"].items():
        lines.append(
            f"| {name} | {metrics['samples']} | {percent(metrics['direct_compatibility_accuracy'])} | "
            f"{percent(metrics['explicit_compatibility_accuracy'])} | "
            f"{percent(metrics['violation_localization_accuracy'])} | {percent(metrics['decision_consistency'])} |"
        )
    lines.extend(["", "## 分约束族", ""])
    lines.append("| 约束族 | N | Direct Compat | Explicit Compat | Localization |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, metrics in summary["by_constraint_family"].items():
        lines.append(
            f"| {name} | {metrics['samples']} | {percent(metrics['direct_compatibility_accuracy'])} | "
            f"{percent(metrics['explicit_compatibility_accuracy'])} | "
            f"{percent(metrics['violation_localization_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 显式协议未参与这些模型的训练，格式有效率与可解析样本语义准确率分开报告。",
            "- Probe 来自已使用过的诊断集，只用于定位瓶颈，不作为新方法的独立最终证据。",
            "- 非法输出不自动修复；仅移除模型生成的固定 `<|im_end|>` 终止标记。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_probe(args: argparse.Namespace) -> None:
    """严格评测两种 Probe 协议并保存逐样本结果。"""
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.metrics import parse_tool_calls

    dataset_dir = args.dataset_dir.resolve()
    predictions_path = args.predictions.resolve()
    samples = load_probe(dataset_dir)
    predictions = read_jsonl(predictions_path)
    prediction_map = {row["id"]: row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise RuntimeError("预测中存在重复 ID")
    if {row["id"] for row in samples} != set(prediction_map):
        raise RuntimeError("预测 ID 与 Probe 数据不一致")

    cases: list[dict[str, Any]] = []
    for sample in samples:
        prediction = prediction_map[sample["id"]]
        direct = parse_tool_calls(prediction["direct_response"])
        direct_compatible = bool(direct.calls) if direct.valid else None
        explicit_valid, explicit_compatible, violation, explicit_error = parse_explicit(
            prediction["explicit_response"]
        )
        expected = sample["expected_compatible"]
        direct_correct = bool(direct.valid and direct_compatible == expected)
        explicit_correct = bool(explicit_valid and explicit_compatible == expected)
        localization_correct = bool(
            not expected and explicit_valid and violation == sample["expected_violation"]
        )
        expected_field: str | None = None
        expected_constraint: str | None = None
        if not expected:
            expected_field, expected_constraint = sample["expected_violation"].rsplit(".", 1)
        violation_field_correct = bool(
            not expected
            and explicit_valid
            and isinstance(violation, str)
            and expected_field is not None
            and expected_field in violation
        )
        violation_constraint_correct = bool(
            not expected
            and explicit_valid
            and isinstance(violation, str)
            and expected_constraint is not None
            and expected_constraint in violation
        )
        cases.append(
            {
                **sample,
                "direct_response": prediction["direct_response"],
                "explicit_response": prediction["explicit_response"],
                "direct_format_valid": direct.valid,
                "direct_parse_error": direct.error,
                "direct_predicted_compatible": direct_compatible,
                "direct_compatibility_correct": direct_correct,
                "explicit_format_valid": explicit_valid,
                "explicit_parse_error": explicit_error,
                "explicit_predicted_compatible": explicit_compatible,
                "explicit_predicted_violation": violation,
                "explicit_compatibility_correct": explicit_correct,
                "violation_localization_correct": localization_correct,
                "violation_field_correct": violation_field_correct,
                "violation_constraint_correct": violation_constraint_correct,
                "decision_consistent": bool(
                    direct.valid and explicit_valid and direct_compatible == explicit_compatible
                ),
            }
        )

    summary = {
        "overall": aggregate(cases),
        "by_sample_type": {
            name: aggregate([case for case in cases if case["sample_type"] == name])
            for name in sorted({case["sample_type"] for case in cases})
        },
        "by_constraint_family": {
            name: aggregate([case for case in cases if case["constraint_family"] == name])
            for name in sorted({case["constraint_family"] for case in cases})
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / f"{args.run_name}_case_results.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    report_path = output_dir / f"{args.run_name}_report.md"
    for path in (case_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(f"评测输出已存在，拒绝覆盖：{path}")
    write_jsonl(case_path, cases)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_probe_report(args.run_name, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    """分派构建、推理或评测子命令。"""
    args = parse_args()
    if args.command == "build":
        build_probe(args)
    elif args.command == "run":
        run_probe(args)
    elif args.command == "evaluate":
        evaluate_probe(args)
    else:
        raise RuntimeError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
