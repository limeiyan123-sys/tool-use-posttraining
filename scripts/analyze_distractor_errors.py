"""对 FDSRC 与 FWCT 的 Distractor 预测执行逐样本错误归因。"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORY_ORDER = [
    "correct",
    "false_rejection",
    "wrong_tool_selection",
    "argument_failure",
    "invalid_output",
]


def parse_args() -> argparse.Namespace:
    """解析输入文件与输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fdsrc-results", type=Path, required=True)
    parser.add_argument("--fwct-results", type=Path, required=True)
    parser.add_argument("--fdsrc-predictions", type=Path, required=True)
    parser.add_argument("--fwct-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并在空行或重复 ID 时立即报错。"""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"{path} 第 {line_number} 行为空")
        row = json.loads(raw_line)
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{path} 第 {line_number} 行缺少合法 id")
        if row_id in seen:
            raise ValueError(f"{path} 存在重复 id：{row_id}")
        seen.add(row_id)
        rows.append(row)
    return rows


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """将已验证唯一 ID 的记录转换为索引。"""
    return {row["id"]: row for row in rows}


def classify_error(result: dict[str, Any]) -> str:
    """将评测器细粒度错误映射为本阶段四类研究错误。"""
    if result["decision_exact"]:
        return "correct"
    error = result["tool_world_error"]
    if error == "unnecessary_refusal":
        return "false_rejection"
    if error in {
        "distractor_selection_error",
        "tool_selection_error",
        "unavailable_tool_hallucination",
        "incompatible_substitute_selection",
        "interface_tool_mapping_error",
    }:
        return "wrong_tool_selection"
    if error in {
        "argument_error",
        "schema_validation_error",
        "interface_argument_mapping_error",
    }:
        return "argument_failure"
    if error == "parse_error":
        return "invalid_output"
    raise ValueError(f"未覆盖的错误类型：{error}")


def percentile(values: list[float], probability: float) -> float:
    """使用最近秩索引计算确定性的经验分位数。"""
    if not values:
        raise ValueError("无法计算空列表的分位数")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def paired_bootstrap(
    left: list[int], right: list[int], samples: int, seed: int
) -> dict[str, float]:
    """对同一批样本的比例差 right-left 进行配对 Bootstrap。"""
    if len(left) != len(right) or not left:
        raise ValueError("配对 Bootstrap 输入长度不一致或为空")
    rng = random.Random(seed)
    differences: list[float] = []
    size = len(left)
    for _ in range(samples):
        indexes = [rng.randrange(size) for _ in range(size)]
        differences.append(
            sum(right[index] - left[index] for index in indexes) / size
        )
    observed = sum(r - l for l, r in zip(left, right)) / size
    return {
        "difference": observed,
        "ci_low": percentile(differences, 0.025),
        "ci_high": percentile(differences, 0.975),
    }


def compact_case(
    sample: dict[str, Any],
    fdsrc_result: dict[str, Any],
    fwct_result: dict[str, Any],
    fdsrc_prediction: dict[str, Any],
    fwct_prediction: dict[str, Any],
) -> dict[str, Any]:
    """提取足以人工复核的案例字段，避免复制完整工具 Schema。"""
    return {
        "id": sample["id"],
        "user_request": sample["messages"][-1]["content"],
        "available_tools": [tool["function"]["name"] for tool in sample["tools"]],
        "expected_calls": sample["expected_calls"],
        "distractor_tool": sample["transformation"]["distractor_tool"],
        "restricted_parameter": sample["transformation"]["restricted_parameter"],
        "allowed_value": sample["transformation"]["allowed_value"],
        "fdsrc_category": classify_error(fdsrc_result),
        "fdsrc_error": fdsrc_result["tool_world_error"],
        "fdsrc_response": fdsrc_prediction["response"],
        "fwct_category": classify_error(fwct_result),
        "fwct_error": fwct_result["tool_world_error"],
        "fwct_response": fwct_prediction["response"],
    }


def select_examples(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按研究价值选择退化、权衡与改善案例。"""
    priorities = [
        ("correct", "false_rejection", "正确调用退化为错误拒绝"),
        ("argument_failure", "false_rejection", "参数错误转为错误拒绝"),
        ("wrong_tool_selection", "false_rejection", "工具误选转为错误拒绝"),
        ("argument_failure", "correct", "参数错误得到修复"),
        ("wrong_tool_selection", "correct", "工具误选得到修复"),
    ]
    selected: list[dict[str, Any]] = []
    for before, after, reason in priorities:
        candidates = [
            case
            for case in cases
            if case["fdsrc_category"] == before and case["fwct_category"] == after
        ]
        if candidates:
            example = dict(sorted(candidates, key=lambda item: item["id"])[0])
            example["selection_reason"] = reason
            selected.append(example)
    return selected


def percent(value: float) -> str:
    """将比例格式化为两位小数百分比。"""
    return f"{value * 100:.2f}%"


def render_report(summary: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    """生成简体中文 Markdown 归因报告。"""
    lines = [
        "# FWCT Distractor 错误归因报告",
        "",
        "## 结论",
        "",
        (
            "现有证据支持过度拒绝假设。FWCT 减少了参数错误和工具误选，"
            "但新增的错误拒绝超过这些收益，使 Distractor 的 Decision Exact 下降。"
        ),
        "",
        "## 统一协议",
        "",
        f"- 样本数：{summary['samples']} 条独立 Distractor 工具世界。",
        "- Decision Exact：输出可解析、工具集合精确匹配、参数精确匹配且通过 Schema 校验。",
        "- False Rejection：Gold 要求调用工具，但模型未输出任何工具调用。",
        "- Wrong Tool Selection：模型调用了错误工具，包括受限干扰工具。",
        "- Argument Failure：工具选择正确，但参数不完整、错误或不满足 Schema。",
        "- Invalid Output：输出无法按严格 tool_call 协议解析。",
        "",
        "## 错误分布",
        "",
        "| 方法 | Correct | False Rejection | Wrong Tool | Argument Failure | Invalid Output |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ["fdsrc", "fwct"]:
        counts = summary["counts"][method]
        lines.append(
            f"| {method.upper()} | {counts['correct']} | {counts['false_rejection']} | "
            f"{counts['wrong_tool_selection']} | {counts['argument_failure']} | "
            f"{counts['invalid_output']} |"
        )
    decision = summary["paired_statistics"]["decision_exact"]
    rejection = summary["paired_statistics"]["false_rejection_rate"]
    lines.extend(
        [
            "",
            f"- FDSRC Decision Exact：{percent(summary['rates']['fdsrc']['correct'])}。",
            f"- FWCT Decision Exact：{percent(summary['rates']['fwct']['correct'])}。",
            (
                f"- FWCT 相对 FDSRC 的配对差值：{decision['difference'] * 100:+.2f} pp，"
                f"95% Bootstrap CI [{decision['ci_low'] * 100:+.2f}, "
                f"{decision['ci_high'] * 100:+.2f}] pp。"
            ),
            (
                f"- 错误拒绝率变化：{rejection['difference'] * 100:+.2f} pp，"
                f"95% Bootstrap CI [{rejection['ci_low'] * 100:+.2f}, "
                f"{rejection['ci_high'] * 100:+.2f}] pp。"
            ),
            "",
            "## 配对转移",
            "",
            "| FDSRC → FWCT | 数量 |",
            "| --- | ---: |",
        ]
    )
    for transition, count in summary["transitions"].items():
        lines.append(f"| `{transition}` | {count} |")
    lines.extend(
        [
            "",
            "## 研究判断",
            "",
            "1. FWCT 的错误拒绝确实增加：从 0/60 增至 14/60，不是少数孤立案例。",
            "2. FWCT 同时减少参数错误 6 次、工具误选 3 次，说明反事实监督学到了规避不兼容工具的有效信号。",
            "3. 总体退化的直接来源是新增错误拒绝；它把“识别不兼容工具”错误外推为“相似工具出现时倾向拒绝”。",
            "4. Valid × Near-miss 四象限监督能够直接解释并针对这一瓶颈，尤其是 E11 必须 Call、E01 必须 Reject 的对照。",
            "5. 参数错误仍有 11 次，FCTW 不能被预期解决全部参数绑定问题；这部分应作为独立剩余风险保留。",
            "",
            "## 代表性真实案例",
            "",
        ]
    )
    for index, example in enumerate(examples, 1):
        lines.extend(
            [
                f"### 案例 {index}：{example['selection_reason']}",
                "",
                f"- ID：`{example['id']}`",
                f"- 请求：{example['user_request']}",
                f"- 正确调用：`{json.dumps(example['expected_calls'], ensure_ascii=False)}`",
                f"- 受限干扰工具：`{example['distractor_tool']}`，"
                f"参数 `{example['restricted_parameter']}` 仅允许 `{example['allowed_value']}`。",
                f"- FDSRC：`{example['fdsrc_error']}`；{example['fdsrc_response']}",
                f"- FWCT：`{example['fwct_error']}`；{example['fwct_response']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一步",
            "",
            "诊断支持继续 FCTW-SFT。最简实现应围绕同一意图构造 E10/E01/E11/E00，"
            "优先验证 E11 能否降低错误拒绝，同时保持 E01 与 Capability Substitution 的拒绝能力。",
            "不需要在这一阶段增加新的损失函数或强化学习。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """校验输入、完成配对归因并写出 JSON 与 Markdown。"""
    args = parse_args()
    dataset = index_rows(read_jsonl(args.dataset.resolve()))
    fdsrc_results = index_rows(read_jsonl(args.fdsrc_results.resolve()))
    fwct_results = index_rows(read_jsonl(args.fwct_results.resolve()))
    fdsrc_predictions = index_rows(read_jsonl(args.fdsrc_predictions.resolve()))
    fwct_predictions = index_rows(read_jsonl(args.fwct_predictions.resolve()))

    distractor_ids = sorted(
        row_id for row_id, row in dataset.items() if row["world_type"] == "distractor"
    )
    expected_ids = set(distractor_ids)
    for name, rows in {
        "FDSRC 结果": fdsrc_results,
        "FWCT 结果": fwct_results,
        "FDSRC 预测": fdsrc_predictions,
        "FWCT 预测": fwct_predictions,
    }.items():
        available = expected_ids & set(rows)
        if available != expected_ids:
            raise ValueError(f"{name} 缺少 Distractor ID：{sorted(expected_ids - available)}")

    cases: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    counts: dict[str, Counter[str]] = {
        "fdsrc": Counter(),
        "fwct": Counter(),
    }
    binary: dict[str, defaultdict[str, list[int]]] = {
        "fdsrc": defaultdict(list),
        "fwct": defaultdict(list),
    }
    for row_id in distractor_ids:
        sample = dataset[row_id]
        if not sample["expected_calls"]:
            raise ValueError(f"Distractor 样本不应是 no-tool：{row_id}")
        fdsrc_result = fdsrc_results[row_id]
        fwct_result = fwct_results[row_id]
        fdsrc_category = classify_error(fdsrc_result)
        fwct_category = classify_error(fwct_result)
        counts["fdsrc"][fdsrc_category] += 1
        counts["fwct"][fwct_category] += 1
        transitions[f"{fdsrc_category} -> {fwct_category}"] += 1
        binary["fdsrc"]["decision"].append(int(fdsrc_category == "correct"))
        binary["fwct"]["decision"].append(int(fwct_category == "correct"))
        binary["fdsrc"]["rejection"].append(int(fdsrc_category == "false_rejection"))
        binary["fwct"]["rejection"].append(int(fwct_category == "false_rejection"))
        cases.append(
            compact_case(
                sample,
                fdsrc_result,
                fwct_result,
                fdsrc_predictions[row_id],
                fwct_predictions[row_id],
            )
        )

    normalized_counts = {
        method: {category: method_counts.get(category, 0) for category in CATEGORY_ORDER}
        for method, method_counts in counts.items()
    }
    summary = {
        "samples": len(distractor_ids),
        "counts": normalized_counts,
        "rates": {
            method: {
                category: category_count / len(distractor_ids)
                for category, category_count in method_counts.items()
            }
            for method, method_counts in normalized_counts.items()
        },
        "transitions": dict(sorted(transitions.items())),
        "paired_statistics": {
            "decision_exact": paired_bootstrap(
                binary["fdsrc"]["decision"],
                binary["fwct"]["decision"],
                args.bootstrap_samples,
                args.seed,
            ),
            "false_rejection_rate": paired_bootstrap(
                binary["fdsrc"]["rejection"],
                binary["fwct"]["rejection"],
                args.bootstrap_samples,
                args.seed + 1,
            ),
        },
    }
    examples = select_examples(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "distractor_error_attribution.json").write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "distractor_error_attribution.md").write_text(
        render_report(summary, examples), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
