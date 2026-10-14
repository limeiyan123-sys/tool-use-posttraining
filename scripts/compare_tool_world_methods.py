#!/usr/bin/env python3
"""对多个工具世界方法进行成对比较、Bootstrap 估计与案例提取。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


WORLD_TYPES = (
    "original",
    "ephemeral_interface",
    "capability_removal",
    "capability_substitution",
    "distractor",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="比较工具世界评测结果")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="格式为 key|显示名|评测目录，可重复指定",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="格式为 reference:candidate，可重复指定",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
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


def parse_method(spec: str, project_root: Path) -> tuple[str, str, Path]:
    """解析并校验方法描述。"""
    parts = spec.split("|", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"非法 method 参数：{spec}")
    key, display_name, raw_path = parts
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return key, display_name, path.resolve()


def load_method(run_key: str, directory: Path) -> dict[str, Any]:
    """读取单个方法的汇总和逐样本结果，并校验 ID 唯一性。"""
    summary_path = directory / f"{run_key}_summary.json"
    cases_path = directory / f"{run_key}_case_results.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(cases_path)
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError(f"{run_key} 的逐样本结果存在重复 ID")
    return {"summary": summary, "rows": rows, "by_id": by_id}


def percentile(sorted_values: list[float], probability: float) -> float:
    """用线性插值计算经验分位数。"""
    if not sorted_values:
        raise ValueError("分位数输入不能为空")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, float | int]:
    """对同一批样本的 Decision Exact 差值执行成对 Bootstrap。"""
    if samples <= 0:
        raise ValueError("bootstrap-samples 必须为正数")
    reference_map = {row["id"]: row for row in reference}
    candidate_map = {row["id"]: row for row in candidate}
    if set(reference_map) != set(candidate_map):
        raise RuntimeError("成对 Bootstrap 的两个方法样本 ID 不一致")
    ids = sorted(reference_map)
    if not ids:
        raise ValueError("成对 Bootstrap 没有样本")
    differences = [
        float(candidate_map[item_id]["decision_exact"])
        - float(reference_map[item_id]["decision_exact"])
        for item_id in ids
    ]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrapped = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in ids) / len(ids)
        for _ in range(samples)
    )
    return {
        "samples": len(ids),
        "delta": observed,
        "ci95_low": percentile(bootstrapped, 0.025),
        "ci95_high": percentile(bootstrapped, 0.975),
    }


def load_samples(frozen_dir: Path) -> dict[str, dict[str, Any]]:
    """依据冻结清单加载原始样本，供真实案例分析使用。"""
    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for filename in manifest["datasets"]:
        rows.extend(read_jsonl(frozen_dir / filename))
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("冻结数据存在重复 ID")
    return by_id


def select_cases(
    reference: dict[str, Any], candidate: dict[str, Any], samples: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """提取真实改善与退化案例，不按文本内容人工挑选。"""
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    for item_id in sorted(reference):
        ref = reference[item_id]
        cand = candidate[item_id]
        bucket: list[dict[str, Any]] | None = None
        if not ref["decision_exact"] and cand["decision_exact"]:
            bucket = improved
        elif ref["decision_exact"] and not cand["decision_exact"]:
            bucket = regressed
        if bucket is None:
            continue
        sample = samples[item_id]
        user_messages = [
            message["content"] for message in sample["messages"] if message.get("role") == "user"
        ]
        bucket.append(
            {
                "id": item_id,
                "world_type": sample["world_type"],
                "user_request": "\n".join(user_messages),
                "transformation": sample["transformation"],
                "expected_calls": sample["expected_calls"],
                "reference_predicted_calls": ref.get("predicted_calls"),
                "reference_error": ref["tool_world_error"],
                "candidate_predicted_calls": cand.get("predicted_calls"),
                "candidate_error": cand["tool_world_error"],
            }
        )
    return {"improved": improved[:2], "regressed": regressed[:1]}


def percent(value: float) -> str:
    """将比例转换为保留两位小数的百分比。"""
    return f"{value * 100:.2f}%"


def render_markdown(result: dict[str, Any]) -> str:
    """渲染中文方法比较报告。"""
    lines = [
        f"# {result['report_name']}",
        "",
        "## 分工具世界结果",
        "",
        "| 方法 | Overall | Original | Ephemeral | Removal | Substitution | Distractor | Interface Equiv. | All-worlds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in result["methods"]:
        summary = method["summary"]
        worlds = summary["by_world"]
        lines.append(
            f"| {method['display_name']} | {percent(summary['overall']['decision_exact'])} | "
            f"{percent(worlds['original']['decision_exact'])} | "
            f"{percent(worlds['ephemeral_interface']['decision_exact'])} | "
            f"{percent(worlds['capability_removal']['decision_exact'])} | "
            f"{percent(worlds['capability_substitution']['decision_exact'])} | "
            f"{percent(worlds['distractor']['decision_exact'])} | "
            f"{percent(summary['interface_equivariance_accuracy'])} | "
            f"{percent(summary['all_worlds_group_accuracy']['accuracy'])} |"
        )
    lines.extend(["", "## 成对 Bootstrap", ""])
    for comparison in result["comparisons"]:
        lines.append(
            f"### {comparison['candidate_display']} 相对 {comparison['reference_display']}"
        )
        lines.extend(
            [
                "",
                "| 范围 | Decision Exact 差值 | 95% CI | 样本数 |",
                "|---|---:|---:|---:|",
            ]
        )
        for scope, stats in comparison["bootstrap"].items():
            lines.append(
                f"| {scope} | {stats['delta'] * 100:+.2f} pp | "
                f"[{stats['ci95_low'] * 100:+.2f}, {stats['ci95_high'] * 100:+.2f}] pp | "
                f"{stats['samples']} |"
            )
        lines.extend(["", "#### 真实案例", ""])
        for label, title in (("improved", "改善"), ("regressed", "退化")):
            cases = comparison["cases"][label]
            if not cases:
                lines.append(f"- {title}：无。")
                continue
            for case in cases:
                lines.append(
                    f"- {title} `{case['id']}`（{case['world_type']}）："
                    f"参考错误 `{case['reference_error']}`，候选错误 `{case['candidate_error']}`。"
                )
                lines.append(f"  - 用户请求：{case['user_request']}")
                lines.append(
                    "  - Gold：`" + json.dumps(case["expected_calls"], ensure_ascii=False) + "`"
                )
                lines.append(
                    "  - 参考预测：`"
                    + json.dumps(case["reference_predicted_calls"], ensure_ascii=False)
                    + "`"
                )
                lines.append(
                    "  - 候选预测：`"
                    + json.dumps(case["candidate_predicted_calls"], ensure_ascii=False)
                    + "`"
                )
        lines.append("")
    lines.extend(
        [
            "## 统计说明",
            "",
            f"- Bootstrap 重采样次数：{result['bootstrap_samples']}。",
            f"- 随机种子：{result['seed']}。",
            "- 区间是样本级成对 Bootstrap 的经验 95% 区间，不对多重比较作校正。",
            "- 非法输出保持失败，不做自动修复。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """比较方法、估计不确定性并保存机器可读和中文报告。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    methods: dict[str, dict[str, Any]] = {}
    for spec in args.method:
        key, display_name, directory = parse_method(spec, project_root)
        if key in methods:
            raise ValueError(f"方法 key 重复：{key}")
        methods[key] = {
            "display_name": display_name,
            **load_method(key, directory),
        }
    id_sets = {key: set(value["by_id"]) for key, value in methods.items()}
    first_ids = next(iter(id_sets.values()))
    for key, ids in id_sets.items():
        if ids != first_ids:
            raise RuntimeError(f"方法 {key} 与其他方法的样本 ID 不一致")

    samples = load_samples(args.frozen_dir.resolve())
    if set(samples) != first_ids:
        raise RuntimeError("冻结样本 ID 与方法结果不一致")
    comparison_results: list[dict[str, Any]] = []
    for comparison_index, spec in enumerate(args.comparison):
        parts = spec.split(":", maxsplit=1)
        if len(parts) != 2 or any(key not in methods for key in parts):
            raise ValueError(f"非法 comparison 参数：{spec}")
        reference_key, candidate_key = parts
        reference = methods[reference_key]
        candidate = methods[candidate_key]
        bootstrap: dict[str, dict[str, float | int]] = {}
        scopes = {"overall": None, **{world: world for world in WORLD_TYPES}}
        for scope_index, (scope, world_type) in enumerate(scopes.items()):
            ref_rows = reference["rows"]
            cand_rows = candidate["rows"]
            if world_type is not None:
                ref_rows = [row for row in ref_rows if row["world_type"] == world_type]
                cand_rows = [row for row in cand_rows if row["world_type"] == world_type]
            bootstrap[scope] = paired_bootstrap(
                ref_rows,
                cand_rows,
                args.bootstrap_samples,
                args.seed + comparison_index * 100 + scope_index,
            )
        comparison_results.append(
            {
                "reference": reference_key,
                "reference_display": reference["display_name"],
                "candidate": candidate_key,
                "candidate_display": candidate["display_name"],
                "bootstrap": bootstrap,
                "cases": select_cases(reference["by_id"], candidate["by_id"], samples),
            }
        )

    result = {
        "report_name": args.report_name,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "methods": [
            {
                "key": key,
                "display_name": value["display_name"],
                "summary": value["summary"],
            }
            for key, value in methods.items()
        ],
        "comparisons": comparison_results,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.report_name}.json"
    markdown_path = output_dir / f"{args.report_name}.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"比较报告已存在，拒绝覆盖：{output_dir}")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
