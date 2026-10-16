#!/usr/bin/env python3
"""构建、运行并评测 Stage 7 历史工具能力依赖实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 20261016
SYSTEM_PROMPT = (
    "你是严谨的工具调用助手。必须依据本轮提供的工具说明选择能够完整满足请求的工具；"
    "不要依赖工具过去的功能。需要调用时只输出结构化工具调用。"
)

FAMILIES = (
    {
        "id": "report_export",
        "names": ("report_node_alpha", "report_node_beta"),
        "capabilities": ("PDF", "CSV"),
        "argument": "report_id",
        "noun": "报表",
        "action": "导出为",
        "prefix": "R",
    },
    {
        "id": "image_convert",
        "names": ("image_node_alpha", "image_node_beta"),
        "capabilities": ("PNG", "JPEG"),
        "argument": "image_id",
        "noun": "图片",
        "action": "转换为",
        "prefix": "IMG",
    },
    {
        "id": "archive_pack",
        "names": ("archive_node_alpha", "archive_node_beta"),
        "capabilities": ("ZIP", "TAR"),
        "argument": "dataset_id",
        "noun": "数据集",
        "action": "打包为",
        "prefix": "DS",
    },
    {
        "id": "document_translate",
        "names": ("language_node_alpha", "language_node_beta"),
        "capabilities": ("英语", "西班牙语"),
        "argument": "document_id",
        "noun": "文档",
        "action": "翻译成",
        "prefix": "DOC",
    },
    {
        "id": "invoice_currency",
        "names": ("finance_node_alpha", "finance_node_beta"),
        "capabilities": ("美元", "欧元"),
        "argument": "invoice_id",
        "noun": "账单",
        "action": "换算为",
        "prefix": "INV",
    },
    {
        "id": "chart_render",
        "names": ("chart_node_alpha", "chart_node_beta"),
        "capabilities": ("柱状图", "折线图"),
        "argument": "table_id",
        "noun": "数据表",
        "action": "渲染为",
        "prefix": "TAB",
    },
)


def parse_args() -> argparse.Namespace:
    """解析子命令和参数。"""
    parser = argparse.ArgumentParser(description="Stage 7 历史工具能力依赖实验")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="构建冻结 Pilot 与受控 SFT 数据")
    build.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="运行确定性工具调用推理")
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--dataset-name", default="pilot_v1.jsonl")
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--adapter-path", type=Path)
    run.add_argument("--output-path", type=Path, required=True)
    run.add_argument("--max-new-tokens", type=int, default=96)
    run.add_argument("--device", choices=("auto", "cpu"), default="auto")
    run.add_argument("--cuda-memory-fraction", type=float, default=0.20)
    run.add_argument("--resume-partial", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="计算配对指标与错误类型")
    evaluate.add_argument("--project-root", type=Path, required=True)
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--dataset-name", default="pilot_v1.jsonl")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """创建 JSONL 文件并拒绝覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    """创建格式化 JSON 文件并拒绝覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def make_tool(name: str, capability: str, family: dict[str, Any]) -> dict[str, Any]:
    """依据能力契约构造名称中立且参数同构的工具。"""
    argument = family["argument"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                f"处理{family['noun']}。当前能力仅支持将{family['noun']}"
                f"{family['action']}{capability}；不支持另一种目标格式或类型。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    argument: {
                        "type": "string",
                        "description": f"待处理{family['noun']}的唯一编号",
                    }
                },
                "required": [argument],
                "additionalProperties": False,
            },
        },
    }


def make_eval_sample(
    family: dict[str, Any], index: int, environment: str, reverse_order: bool = False
) -> dict[str, Any]:
    """构造一个 Original 或 Updated 配对评测样本。"""
    if environment not in {"original", "updated"}:
        raise ValueError(f"未知环境：{environment}")
    requested_index = index % 2
    capability = family["capabilities"][requested_index]
    item_id = f"{family['prefix']}{index + 1:03d}"
    original_capabilities = family["capabilities"]
    current_capabilities = (
        original_capabilities
        if environment == "original"
        else tuple(reversed(original_capabilities))
    )
    tools = [
        make_tool(name, current_capabilities[position], family)
        for position, name in enumerate(family["names"])
    ]
    if reverse_order:
        tools.reverse()
    correct_position = current_capabilities.index(capability)
    stale_position = original_capabilities.index(capability)
    base_id = f"{family['id']}__test_{index + 1:02d}"
    return {
        "id": f"{base_id}__{environment}" + ("__order_swapped" if reverse_order else ""),
        "base_id": base_id,
        "family": family["id"],
        "environment": environment,
        "order_swapped": reverse_order,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"请将编号 {item_id} 的{family['noun']}"
                    f"{family['action']}{capability}。"
                ),
            }
        ],
        "tools": tools,
        "required_capability": capability,
        "tool_capabilities": dict(zip(family["names"], current_capabilities)),
        "original_tool_capabilities": dict(zip(family["names"], original_capabilities)),
        "expected_call": {
            "name": family["names"][correct_position],
            "arguments": {family["argument"]: item_id},
        },
        "stale_tool": family["names"][stale_position],
    }


def to_sft_record(family: dict[str, Any], index: int) -> dict[str, Any]:
    """构造只暴露 Original 能力关系的独立 SFT 样本。"""
    requested_index = index % 2
    capability = family["capabilities"][requested_index]
    item_id = f"{family['prefix']}TR{index + 1:03d}"
    tools = [
        make_tool(name, family["capabilities"][position], family)["function"]
        for position, name in enumerate(family["names"])
    ]
    call = {
        "name": family["names"][requested_index],
        "arguments": {family["argument"]: item_id},
    }
    return {
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"请处理编号 {item_id} 的{family['noun']}，目标是"
                    f"{family['action']}{capability}。"
                ),
            },
            {"from": "function_call", "value": json.dumps(call, ensure_ascii=False)},
        ],
        "tools": json.dumps(tools, ensure_ascii=False),
    }


def build_data(args: argparse.Namespace) -> None:
    """生成冻结 Pilot、顺序控制与独立受控训练数据。"""
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)

    pilot: list[dict[str, Any]] = []
    order_swapped: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {}
    for family in FAMILIES:
        contracts[family["id"]] = {
            "tool_names": list(family["names"]),
            "original_capabilities": dict(zip(family["names"], family["capabilities"])),
            "argument": family["argument"],
        }
        for index in range(4):
            for environment in ("original", "updated"):
                pilot.append(make_eval_sample(family, index, environment))
                order_swapped.append(make_eval_sample(family, index, environment, True))
        for index in range(8):
            train.append(to_sft_record(family, index))
        for index in range(8, 10):
            dev.append(to_sft_record(family, index))

    paths = {
        "pilot_v1.jsonl": pilot,
        "pilot_order_swapped_v1.jsonl": order_swapped,
    }
    for filename, rows in paths.items():
        write_jsonl(output_dir / filename, rows)
    write_json(output_dir / "contracts.json", contracts)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "dev.json", dev)
    dataset_info = {
        "stage7_history_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "stage7_history_dev": {
            "file_name": "dev.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
    }
    write_json(output_dir / "dataset_info.json", dataset_info)
    manifest = {
        "version": "stage7_historical_dependency_v1",
        "seed": SEED,
        "families": len(FAMILIES),
        "base_test_tasks": len(pilot) // 2,
        "pilot_samples": len(pilot),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "test_requests_disjoint_from_training": True,
        "gold_rule": "当前工具描述中的能力必须与用户目标完全一致",
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_dataset(dataset_dir: Path, dataset_name: str) -> list[dict[str, Any]]:
    """依据冻结清单加载并校验评测数据。"""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    path = dataset_dir / dataset_name
    metadata = manifest["files"].get(dataset_name)
    if metadata is None or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"冻结数据哈希不匹配：{path}")
    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("评测样本 ID 重复")
    return rows


def prepare_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """加入固定系统指令且不改写用户请求。"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *sample["messages"]]


def run_inference(args: argparse.Namespace) -> None:
    """加载基础模型或 LoRA，并执行确定性推理。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_dir = args.dataset_dir.resolve()
    rows = load_dataset(dataset_dir, args.dataset_name)
    model_path = args.model_path.resolve()
    adapter_path = args.adapter_path.resolve() if args.adapter_path else None
    output_path = args.output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"预测文件已存在，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    existing: dict[str, dict[str, Any]] = {}
    mode = "x"
    if partial_path.exists():
        if not args.resume_partial:
            raise FileExistsError(f"发现临时结果，需显式续跑：{partial_path}")
        partial_rows = read_jsonl(partial_path)
        existing = {row["id"]: row for row in partial_rows}
        if len(existing) != len(partial_rows):
            raise RuntimeError("临时结果包含重复 ID")
        mode = "a"
        print(f"[WARNING] 续跑已完成的 {len(existing)} 条预测")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(SEED)
    if args.device != "cpu" and torch.cuda.is_available():
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError("cuda-memory-fraction 必须位于 (0, 1] 区间")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        device_map=None if args.device == "cpu" else "auto",
        local_files_only=True,
    )
    if args.device == "cpu":
        base_model.to("cpu")
    model = (
        PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
        if adapter_path
        else base_model
    )
    model.eval()
    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with partial_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(rows, start=1):
            if sample["id"] in existing:
                continue
            prompt = tokenizer.apply_chat_template(
                prepare_messages(sample),
                tools=sample["tools"],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            sample_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            record = {
                "id": sample["id"],
                "response": tokenizer.decode(new_tokens, skip_special_tokens=False).strip(),
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(new_tokens.shape[0]),
                "latency_seconds": round(time.perf_counter() - sample_started, 6),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(f"已完成 {index}/{len(rows)}")
    partial_path.replace(output_path)
    metadata = {
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "dataset": args.dataset_name,
        "dataset_sha256": sha256_file(dataset_dir / args.dataset_name),
        "seed": SEED,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "samples": len(rows),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "device": args.device,
        "cuda_memory_fraction": args.cuda_memory_fraction if args.device != "cpu" else None,
        "peak_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            if torch.cuda.is_available() and args.device != "cpu"
            else 0.0
        ),
        "prediction_sha256": sha256_file(output_path),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def classify_case(sample: dict[str, Any], prediction: dict[str, Any], parse_tool_calls: Any) -> dict[str, Any]:
    """执行 Mock Tool 契约并给出可验证错误类型。"""
    response = prediction.get("response")
    if not isinstance(response, str):
        raise TypeError(f"预测 {sample['id']} 缺少字符串 response")
    parsed = parse_tool_calls(response)
    expected = sample["expected_call"]
    if not parsed.valid:
        error_type = "invalid_output"
        selected_tool = None
        arguments_correct = False
        correct = False
    elif len(parsed.calls) == 0:
        error_type = "false_reject"
        selected_tool = None
        arguments_correct = False
        correct = False
    elif len(parsed.calls) != 1:
        error_type = "invalid_output"
        selected_tool = None
        arguments_correct = False
        correct = False
    else:
        call = parsed.calls[0]
        selected_tool = call["name"]
        arguments_correct = call["arguments"] == expected["arguments"]
        if selected_tool != expected["name"]:
            error_type = "semantic_capability_error"
            correct = False
        elif not arguments_correct:
            error_type = "argument_error"
            correct = False
        else:
            error_type = "none"
            correct = True
    stale_choice = bool(
        sample["environment"] == "updated"
        and selected_tool is not None
        and selected_tool == sample["stale_tool"]
        and selected_tool != expected["name"]
    )
    return {
        "id": sample["id"],
        "base_id": sample["base_id"],
        "family": sample["family"],
        "environment": sample["environment"],
        "order_swapped": sample["order_swapped"],
        "response": response,
        "parse_valid": parsed.valid,
        "parse_error": parsed.error,
        "selected_tool": selected_tool,
        "expected_tool": expected["name"],
        "stale_tool": sample["stale_tool"],
        "arguments_correct": arguments_correct,
        "correct": correct,
        "stale_choice": stale_choice,
        "semantic_error": error_type == "semantic_capability_error",
        "false_reject": error_type == "false_reject",
        "error_type": error_type,
        "mock_execution": {
            "success": correct,
            "required_capability": sample["required_capability"],
            "selected_capability": sample["tool_capabilities"].get(selected_tool),
        },
    }


def mean(values: list[bool | float]) -> float:
    """计算非空序列均值。"""
    if not values:
        raise ValueError("不能计算空序列均值")
    return sum(values) / len(values)


def percentile(values: list[float], probability: float) -> float:
    """以最近秩规则计算 Bootstrap 分位数。"""
    if not values:
        raise ValueError("不能计算空序列分位数")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def paired_bootstrap(pairs: list[dict[str, Any]], samples: int) -> dict[str, list[float]]:
    """按基础任务重采样，估计配对指标的 95% 区间。"""
    rng = random.Random(SEED)
    metrics: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        selected = [rng.choice(pairs) for _ in pairs]
        original = [pair["original"]["correct"] for pair in selected]
        updated = [pair["updated"]["correct"] for pair in selected]
        metrics["original_accuracy"].append(mean(original))
        metrics["updated_accuracy"].append(mean(updated))
        metrics["updated_minus_original"].append(mean(updated) - mean(original))
        metrics["pair_success_rate"].append(
            mean([left and right for left, right in zip(original, updated)])
        )
        metrics["stale_choice_rate_all_updated"].append(
            mean([pair["updated"]["stale_choice"] for pair in selected])
        )
    return {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in metrics.items()
    }


def aggregate_cases(cases: list[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    """聚合核心指标、家族结果和配对不确定性。"""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["base_id"]][case["environment"]] = case
    if any(set(pair) != {"original", "updated"} for pair in grouped.values()):
        raise RuntimeError("Original/Updated 配对不完整")
    pairs = list(grouped.values())
    original = [pair["original"] for pair in pairs]
    updated = [pair["updated"] for pair in pairs]
    original_correct = [case["correct"] for case in original]
    updated_correct = [case["correct"] for case in updated]
    original_correct_pairs = [pair for pair in pairs if pair["original"]["correct"]]
    summary = {
        "base_tasks": len(pairs),
        "original_accuracy": mean(original_correct),
        "updated_accuracy": mean(updated_correct),
        "updated_minus_original": mean(updated_correct) - mean(original_correct),
        "pair_success_rate": mean(
            [left and right for left, right in zip(original_correct, updated_correct)]
        ),
        "stale_choice_rate_all_updated": mean([case["stale_choice"] for case in updated]),
        "stale_choice_all_updated_numerator": sum(case["stale_choice"] for case in updated),
        "stale_choice_all_updated_denominator": len(updated),
        "stale_choice_rate_given_original_correct": (
            mean([pair["updated"]["stale_choice"] for pair in original_correct_pairs])
            if original_correct_pairs
            else None
        ),
        "stale_choice_given_original_correct_numerator": sum(
            pair["updated"]["stale_choice"] for pair in original_correct_pairs
        ),
        "stale_choice_given_original_correct_denominator": len(original_correct_pairs),
        "semantic_error_rate": mean([case["semantic_error"] for case in cases]),
        "false_reject_rate": mean([case["false_reject"] for case in cases]),
        "error_counts": dict(sorted(Counter(case["error_type"] for case in cases).items())),
        "bootstrap_95_ci": paired_bootstrap(pairs, bootstrap_samples),
    }
    by_family: dict[str, Any] = {}
    for family in sorted({case["family"] for case in cases}):
        family_cases = [case for case in cases if case["family"] == family]
        family_original = [case for case in family_cases if case["environment"] == "original"]
        family_updated = [case for case in family_cases if case["environment"] == "updated"]
        by_family[family] = {
            "original_accuracy": mean([case["correct"] for case in family_original]),
            "updated_accuracy": mean([case["correct"] for case in family_updated]),
            "stale_choice_rate_all_updated": mean(
                [case["stale_choice"] for case in family_updated]
            ),
        }
    summary["by_family"] = by_family
    return summary


def percent(value: float | None) -> str:
    """格式化可空百分比。"""
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_report(run_name: str, summary: dict[str, Any]) -> str:
    """渲染单模型 Stage 7 指标报告。"""
    ci = summary["bootstrap_95_ci"]
    lines = [
        f"# Stage 7 {run_name} 配对评测",
        "",
        f"- 基础任务数：{summary['base_tasks']}",
        f"- Original Accuracy：{percent(summary['original_accuracy'])}",
        f"- Updated Accuracy：{percent(summary['updated_accuracy'])}",
        f"- Updated - Original：{summary['updated_minus_original'] * 100:+.2f} pp",
        f"- Pair Success Rate：{percent(summary['pair_success_rate'])}",
        (
            "- Stale-Choice Rate（全部 Updated）："
            f"{summary['stale_choice_all_updated_numerator']}/"
            f"{summary['stale_choice_all_updated_denominator']} = "
            f"{percent(summary['stale_choice_rate_all_updated'])}"
        ),
        (
            "- Stale-Choice Rate（Original 正确条件下）："
            f"{summary['stale_choice_given_original_correct_numerator']}/"
            f"{summary['stale_choice_given_original_correct_denominator']} = "
            f"{percent(summary['stale_choice_rate_given_original_correct'])}"
        ),
        f"- Semantic Error Rate：{percent(summary['semantic_error_rate'])}",
        f"- False Reject Rate：{percent(summary['false_reject_rate'])}",
        "",
        "## 配对 Bootstrap 95% 区间",
        "",
        "| 指标 | 下界 | 上界 |",
        "|---|---:|---:|",
    ]
    for key in (
        "original_accuracy",
        "updated_accuracy",
        "updated_minus_original",
        "pair_success_rate",
        "stale_choice_rate_all_updated",
    ):
        lines.append(f"| {key} | {percent(ci[key][0])} | {percent(ci[key][1])} |")
    lines.extend(["", "## 分工具家族", "", "| 家族 | Original | Updated | Stale Choice |", "|---|---:|---:|---:|"])
    for family, metrics in summary["by_family"].items():
        lines.append(
            f"| {family} | {percent(metrics['original_accuracy'])} | "
            f"{percent(metrics['updated_accuracy'])} | "
            f"{percent(metrics['stale_choice_rate_all_updated'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- Pilot 只有 24 个配对任务，Bootstrap 区间用于表达不确定性，不宣称统计显著。",
            "- Semantic Error 仅指格式合法、参数可比较但工具能力不满足请求的调用。",
            "- Mock Tool 依据冻结能力契约执行，不使用模型自由判断兼容性。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> None:
    """严格评测预测并保存逐样本结果、指标与报告。"""
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.metrics import parse_tool_calls

    dataset_dir = args.dataset_dir.resolve()
    samples = load_dataset(dataset_dir, args.dataset_name)
    predictions = read_jsonl(args.predictions.resolve())
    prediction_map = {row["id"]: row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise RuntimeError("预测中存在重复 ID")
    if {row["id"] for row in samples} != set(prediction_map):
        raise RuntimeError("预测 ID 与冻结数据不一致")
    cases = [classify_case(sample, prediction_map[sample["id"]], parse_tool_calls) for sample in samples]
    summary = aggregate_cases(cases, args.bootstrap_samples)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / f"{args.run_name}_cases.jsonl", cases)
    write_json(output_dir / f"{args.run_name}_summary.json", summary)
    report_path = output_dir / f"{args.run_name}_report.md"
    report_path.write_text(render_report(args.run_name, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    """分派 Stage 7 子命令。"""
    args = parse_args()
    if args.command == "build":
        build_data(args)
    elif args.command == "run":
        run_inference(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        raise RuntimeError(f"未处理的子命令：{args.command}")


if __name__ == "__main__":
    main()
