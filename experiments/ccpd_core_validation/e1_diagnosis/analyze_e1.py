#!/usr/bin/env python3
"""统一分析 E1 原生工具调用与受控决策探针。"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable


MODEL_FILES = {
    "base": "zero_shot_predictions.jsonl",
    "fdsrc": "fdsrc_predictions.jsonl",
    "fwct": "fwct_predictions.jsonl",
    "fctw": "fctw_predictions.jsonl",
}


def parse_args() -> argparse.Namespace:
    """解析项目、探针结果与输出路径。"""
    parser = argparse.ArgumentParser(description="分析 CCPD E1 合同敏感性")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
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


def percentile(values: list[float], fraction: float) -> float:
    """使用线性插值计算百分位数。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    repeats: int,
    seed: int,
) -> list[float]:
    """以基础请求为单位执行非参数 Bootstrap。"""
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repeats):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        estimates.append(statistic(sampled))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def mean_flag(rows: list[dict[str, Any]], key: str) -> float:
    """计算布尔或数值字段均值。"""
    return statistics.fmean(float(row[key]) for row in rows)


def metric(
    rows: list[dict[str, Any]], key: str, repeats: int, seed: int
) -> dict[str, Any]:
    """计算均值及任务级 Bootstrap 区间。"""
    estimate = mean_flag(rows, key)
    ci = bootstrap_ci(rows, lambda sample: mean_flag(sample, key), repeats, seed)
    return {"value": estimate, "ci95": ci}


def parse_native_decision(text: str, parse_tool_calls: Callable[[str], Any]) -> str:
    """严格解析原生工具调用，不把格式错误回退为拒绝。"""
    parsed = parse_tool_calls(text)
    if not parsed.valid:
        return "INVALID"
    return "CALL" if parsed.calls else "DECLINE"


def make_pair_rows(
    pairs: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    parse_tool_calls: Callable[[str], Any],
) -> list[dict[str, Any]]:
    """把两类协议按基础请求和合同侧严格对齐。"""
    native = {row["id"]: row for row in native_rows}
    scores = {(row["base_id"], row["side"]): row for row in score_rows}
    if len(native) != len(native_rows) or len(scores) != len(score_rows):
        raise RuntimeError("预测中存在重复键")
    output: list[dict[str, Any]] = []
    for pair in pairs:
        base_id = pair["base_id"]
        valid_native = parse_native_decision(
            native[pair["valid_sample"]["id"]]["direct_response"], parse_tool_calls
        )
        invalid_native = parse_native_decision(
            native[pair["invalid_sample"]["id"]]["direct_response"], parse_tool_calls
        )
        valid_score = scores[(base_id, "valid")]
        invalid_score = scores[(base_id, "invalid")]
        score_plus = float(valid_score["decision_score"])
        score_minus = float(invalid_score["decision_score"])
        valid_forward = float(valid_score["forward_call_score"])
        invalid_forward = float(invalid_score["forward_call_score"])
        valid_reverse = float(valid_score["reverse_call_score"])
        invalid_reverse = float(invalid_score["reverse_call_score"])
        probe_valid = valid_score["predicted_decision"]
        probe_invalid = invalid_score["predicted_decision"]
        output.append(
            {
                "base_id": base_id,
                "audit_tier": pair["audit_tier"],
                "constraint_family": pair["constraint_family"],
                "native_valid_decision": valid_native,
                "native_invalid_decision": invalid_native,
                "native_pair_success": valid_native == "CALL" and invalid_native == "DECLINE",
                "native_valid_call": valid_native == "CALL",
                "native_invalid_call": invalid_native == "CALL",
                "native_false_reject": valid_native == "DECLINE",
                "native_valid_invalid_output": valid_native == "INVALID",
                "native_invalid_invalid_output": invalid_native == "INVALID",
                "probe_valid_decision": probe_valid,
                "probe_invalid_decision": probe_invalid,
                "probe_pair_success": probe_valid == "CALL" and probe_invalid == "DECLINE",
                "probe_valid_call": probe_valid == "CALL",
                "probe_invalid_call": probe_invalid == "CALL",
                "probe_false_reject": probe_valid == "DECLINE",
                "probe_valid_tie": probe_valid == "TIE",
                "probe_invalid_tie": probe_invalid == "TIE",
                "probe_concordant": score_plus > score_minus,
                "probe_concordance_tie": score_plus == score_minus,
                "probe_forward_concordant": valid_forward > invalid_forward,
                "probe_reverse_concordant": valid_reverse > invalid_reverse,
                "probe_score_plus": score_plus,
                "probe_score_minus": score_minus,
                "probe_score_difference": score_plus - score_minus,
                "probe_mean_decision_score": (score_plus + score_minus) / 2.0,
                "probe_forward_score_difference": valid_forward - invalid_forward,
                "probe_reverse_score_difference": valid_reverse - invalid_reverse,
                "probe_label_order_shift_valid": valid_forward - valid_reverse,
                "probe_label_order_shift_invalid": invalid_forward - invalid_reverse,
                "probe_label_order_agreement_valid": (
                    (valid_score["forward_call_score"] > 0) == (valid_score["reverse_call_score"] > 0)
                ),
                "probe_label_order_agreement_invalid": (
                    (invalid_score["forward_call_score"] > 0)
                    == (invalid_score["reverse_call_score"] > 0)
                ),
                "native_probe_agreement_valid": valid_native == probe_valid,
                "native_probe_agreement_invalid": invalid_native == probe_invalid,
            }
        )
    return output


def summarize(rows: list[dict[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    """计算 E1 预注册的全部配对指标。"""
    keys = [
        "native_pair_success",
        "native_valid_call",
        "native_invalid_call",
        "native_false_reject",
        "native_valid_invalid_output",
        "native_invalid_invalid_output",
        "probe_pair_success",
        "probe_valid_call",
        "probe_invalid_call",
        "probe_false_reject",
        "probe_valid_tie",
        "probe_invalid_tie",
        "probe_concordant",
        "probe_concordance_tie",
        "probe_forward_concordant",
        "probe_reverse_concordant",
        "probe_score_difference",
        "probe_mean_decision_score",
        "probe_forward_score_difference",
        "probe_reverse_score_difference",
        "probe_label_order_shift_valid",
        "probe_label_order_shift_invalid",
        "probe_label_order_agreement_valid",
        "probe_label_order_agreement_invalid",
        "native_probe_agreement_valid",
        "native_probe_agreement_invalid",
    ]
    return {
        "pairs": len(rows),
        **{key: metric(rows, key, repeats, seed + index) for index, key in enumerate(keys)},
    }


def paired_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """计算两个模型在相同基础任务上的配对指标差。"""
    left_map = {row["base_id"]: row for row in left}
    right_map = {row["base_id"]: row for row in right}
    if set(left_map) != set(right_map):
        raise RuntimeError("模型间基础任务不一致")
    differences = [
        {"difference": float(right_map[base_id][key]) - float(left_map[base_id][key])}
        for base_id in sorted(left_map)
    ]
    estimate = mean_flag(differences, "difference")
    ci = bootstrap_ci(
        differences, lambda sample: mean_flag(sample, "difference"), repeats, seed
    )
    return {"right_minus_left": estimate, "ci95": ci}


def main() -> None:
    """对齐四个模型并保存逐对结果、汇总与 FWCT-FDSRC 配对差。"""
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from evaluation.metrics import parse_tool_calls

    e1_dir = root / "experiments/ccpd_core_validation/e1_diagnosis"
    all_pairs = read_jsonl(e1_dir / "data/pair_audit.jsonl")
    native_dir = root / "reports/stage6_schema_grounded/capability_probe"
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 E1 汇总目录：{output}")
    output.mkdir(parents=True)
    model_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for model_name, native_file in MODEL_FILES.items():
        rows = make_pair_rows(
            all_pairs,
            read_jsonl(native_dir / native_file),
            read_jsonl(args.score_dir.resolve() / f"{model_name}_scores.jsonl"),
            parse_tool_calls,
        )
        model_rows[model_name] = rows
        with (output / f"{model_name}_paired_cases.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        strict = [row for row in rows if row["audit_tier"] == "strict_required"]
        summaries[model_name] = {
            "strict_primary": summarize(strict, args.bootstrap_repeats, args.seed),
            "all_30_sensitivity": summarize(rows, args.bootstrap_repeats, args.seed),
        }

    comparison_keys = [
        "native_pair_success",
        "native_valid_call",
        "native_invalid_call",
        "probe_pair_success",
        "probe_concordant",
        "probe_score_difference",
        "probe_mean_decision_score",
    ]
    fdsrc = [row for row in model_rows["fdsrc"] if row["audit_tier"] == "strict_required"]
    fwct = [row for row in model_rows["fwct"] if row["audit_tier"] == "strict_required"]
    comparisons = {
        "fwct_minus_fdsrc_strict": {
            key: paired_difference(fdsrc, fwct, key, args.bootstrap_repeats, args.seed + index)
            for index, key in enumerate(comparison_keys)
        }
    }
    result = {
        "bootstrap": {"unit": "base_request", "repeats": args.bootstrap_repeats, "seed": args.seed},
        "models": summaries,
        "comparisons": comparisons,
    }
    (output / "e1_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
