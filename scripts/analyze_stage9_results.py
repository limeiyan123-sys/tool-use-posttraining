#!/usr/bin/env python3
"""汇总 Stage 9 的类型错误、配对不确定性、字段识别和免训练方法证据。"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from stage9_schema_binding import CONDITIONS, SEED, percentile, read_jsonl, sha256_file, write_json


def paired_interval(groups, statistic, draws=5000):
    """对预先分好的任务或模板簇整体重采样，保留簇内相关性。"""
    rng = random.Random(SEED)
    values = [statistic([row for _ in groups for row in rng.choice(groups)]) for _ in range(draws)]
    return [percentile(values, 0.025), percentile(values, 0.975)]


def effect(rows, metric, contrast):
    """计算配对对比；参数准确率按实际用户要求字段数加权。"""
    total = 0.0
    for condition, weight in contrast.items():
        selected = [row for row in rows if row["condition"] == condition]
        if metric == "binding":
            value = sum(row["correct_fields"] for row in selected) / sum(row["expected_field_count"] for row in selected)
        else:
            value = sum(row["decision_exact"] for row in selected) / len(selected)
        total += value * weight
    return total


def confidence_stats(cases):
    """同时报告任务级与六模板级 Bootstrap，防止伪独立解释。"""
    groups = {}
    for key in ("base_id", "family"):
        grouped = defaultdict(list)
        for case in cases:
            grouped[case[key]].append(case)
        groups[key] = list(grouped.values())
    contrasts = {"B-A": {"B": 1, "A": -1}, "C-A": {"C": 1, "A": -1}, "D-A": {"D": 1, "A": -1}, "interaction": {"D": 1, "C": -1, "B": -1, "A": 1}}
    result = {}
    for metric in ("decision", "binding"):
        for name, contrast in contrasts.items():
            statistic = lambda rows, m=metric, c=contrast: effect(rows, m, c)
            result[f"{metric}:{name}"] = {"estimate": statistic(cases), "task_ci": paired_interval(groups["base_id"], statistic), "family_ci": paired_interval(groups["family"], statistic)}
    return result


def detailed_counts(cases):
    """按字段类型、可选性和错误种类提供分子分母。"""
    result = {}
    for condition in sorted({case["condition"] for case in cases}):
        rows = [case for case in cases if case["condition"] == condition]
        by_type = defaultdict(lambda: {"correct": 0, "total": 0, "omitted": 0})
        for row in rows:
            for field in row["expected_arguments"]:
                entry = by_type[row["field_types"][field]]
                entry["total"] += 1
                entry["omitted"] += field in row["omitted_fields"]
                entry["correct"] += row["tool_correct"] and field not in row["omitted_fields"] + row["wrong_value_fields"]
        result[condition] = {
            "by_type": dict(by_type),
            "extra_argument_samples": sum(bool(row["extra_fields"]) for row in rows),
            "required_binding_perfect_samples": sum(row["correct_fields"] == row["expected_field_count"] and row["tool_correct"] for row in rows),
            "same_type_swaps": sum(bool(row["same_type_swaps"]) for row in rows),
            "cross_type_misbindings": sum(bool(row["cross_type_misbinding_pairs"]) for row in rows),
            "omitted_optional": sum(len(row["optional_omitted"]) for row in rows),
        }
    return result


def baseline_comparison(direct, baseline):
    """在完全相同 B/D 任务上计算基线改善、退化及配对区间。"""
    reference = {case["id"]: case for case in direct}
    pairs = []
    for case in baseline:
        ref = reference[case["id"]]
        pairs.append({"base_id": case["base_id"], "condition": case["condition"], "direct": ref, "baseline": case})
    result = {}
    for condition in ("B", "D", "BD"):
        selected = [pair for pair in pairs if condition == "BD" or pair["condition"] == condition]
        groups = defaultdict(list)
        for pair in selected:
            groups[pair["base_id"]].append(pair)
        statistic = lambda rows: sum(int(row["baseline"]["decision_exact"]) - int(row["direct"]["decision_exact"]) for row in rows) / len(rows)
        result[condition] = {"samples": len(selected), "decision_delta": statistic(selected), "paired_task_ci": paired_interval(list(groups.values()), statistic), "recoveries": sum(not row["direct"]["decision_exact"] and row["baseline"]["decision_exact"] for row in selected), "regressions": sum(row["direct"]["decision_exact"] and not row["baseline"]["decision_exact"] for row in selected)}
    return result


def field_results(root, case_maps):
    """用同一目标字段比较字段识别与已有完整调用，不修复原始输出。"""
    dataset = root / "data/evaluation/stage9_field_probe_v1/field_probe.jsonl"
    if not dataset.exists():
        return {"executed": False, "reason": "未生成字段识别数据；详见阶段报告的执行条件"}
    targets = {row["id"]: row for row in read_jsonl(dataset)}
    results = {}
    for path in sorted((root / "reports/stage9_schema_binding/predictions").glob("*_field_probe_v1.jsonl")):
        model = path.name.removesuffix("_field_probe_v1.jsonl")
        rows = []
        for pred in read_jsonl(path):
            target = targets[pred["id"]]
            response = pred["response"].removesuffix("<|im_end|>").strip()
            field = target["probe_target_semantic"]
            source_id = pred["id"].removesuffix("_field_probe")
            source = case_maps[model][source_id]
            full_correct = source["tool_correct"] and field not in source["omitted_fields"] + source["wrong_value_fields"]
            rows.append({"id": pred["id"], "group": target["probe_group"], "condition": target["condition"], "target_field": field, "recognition_correct": response == target["probe_target_actual"], "recognized": response, "expected": target["probe_target_actual"], "full_call_field_correct": full_correct, "full_call_decision_correct": source["decision_exact"], "source_tool_correct": source["tool_correct"], "source_error_type": source["error_type"]})
        results[model] = {"cases": rows, "by_group_condition": {f"{group}:{condition}": {"n": len(selected), "recognition_correct": sum(row["recognition_correct"] for row in selected), "full_field_correct": sum(row["full_call_field_correct"] for row in selected), "recognition_only_success": sum(row["recognition_correct"] and not row["full_call_field_correct"] for row in selected), "full_only_success": sum(not row["recognition_correct"] and row["full_call_field_correct"] for row in selected)} for group in ("regression", "control") for condition in CONDITIONS if (selected := [row for row in rows if row["group"] == group and row["condition"] == condition])}}
    return {"executed": bool(results), "selection_manifest": json.loads((dataset.parent / "manifest.json").read_text()), "models": results}


def main():
    """读取统一评分结果并输出可复核的离线证据汇总。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    case_lists = {path.name.removesuffix("_cases.jsonl"): read_jsonl(path) for path in sorted(args.result_dir.glob("*_cases.jsonl"))}
    direct = {name: cases for name, cases in case_lists.items() if name in {"base", "fdsrc", "fctw"}}
    result = {"confidence": {name: confidence_stats(cases) for name, cases in direct.items()}, "details": {name: detailed_counts(cases) for name, cases in case_lists.items()}, "baseline_comparisons": {}, "field_protocol": field_results(args.root, {name: {row["id"]: row for row in cases} for name, cases in direct.items()})}
    for name, cases in case_lists.items():
        if name not in direct:
            model = name.split("_", 1)[0]
            result["baseline_comparisons"][name] = baseline_comparison(direct[model], cases)
    token_rows = read_jsonl(args.root / "data/evaluation/stage9_schema_binding_v2/identifier_tokenization.jsonl")
    result["tokenization"] = {condition: {"mean": sum(values) / len(values), "min": min(values), "max": max(values)} for condition in ("A", "B") if (values := [row["token_count"] for row in token_rows if row["condition"] == condition])}
    metadata = [json.loads(path.read_text()) for path in (args.root / "reports/stage9_schema_binding/predictions").glob("*.metadata.json")]
    result["resource"] = {"inference_samples_including_invalid_v1": sum(row["samples"] for row in metadata), "inference_seconds_including_invalid_v1": sum(row["runtime_seconds"] for row in metadata), "maximum_peak_memory_gib": max(row["peak_memory_gib"] for row in metadata), "training_runs": 0, "paid_api_calls": 0, "currency_cost": "共享课题组设备，无已知新增付费；无电费或折旧账单，不能推算人民币成本"}
    result["sources"] = {str(path.relative_to(args.root)): sha256_file(path) for path in args.result_dir.glob("*_cases.jsonl")}
    write_json(args.output, result)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        summaries = json.loads((args.result_dir / "comparison.json").read_text())
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        colors = {"base": "#3178c6", "fdsrc": "#29816d", "fctw": "#ba6046"}
        for axis, metric, title in zip(axes, ["decision_exact", "parameter_binding_accuracy"], ["Decision Exact", "Parameter Binding Accuracy"]):
            for index, (name, color) in enumerate(colors.items()):
                values = [100 * summaries[name]["by_condition"][condition][metric] for condition in CONDITIONS]
                positions = [position + (index - 1) * 0.24 for position in range(4)]
                axis.bar(positions, values, width=0.23, label=name.upper(), color=color)
            axis.set_xticks(range(4), CONDITIONS)
            axis.set_ylim(0, 105)
            axis.set_ylabel("%")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        axes[0].legend(loc="lower left", ncol=3, fontsize=8)
        figure.suptitle("Stage 9 | 24 paired tasks, 6 tool templates")
        figure.savefig(args.output.parent / "factorial_results.png", dpi=180)
        plt.close(figure)
    print(json.dumps({"tokenization": result["tokenization"], "resource": result["resource"], "baseline_comparisons": result["baseline_comparisons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
