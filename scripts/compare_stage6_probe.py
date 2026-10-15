#!/usr/bin/env python3
"""汇总 Stage 6 单工具 Probe，并据证据生成方法选择报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = ("zero_shot", "lora", "fdsrc", "fwct", "fctw")
DISPLAY_NAMES = {
    "zero_shot": "Zero-shot",
    "lora": "LoRA",
    "fdsrc": "FDSRC",
    "fwct": "FWCT",
    "fctw": "FCTW",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="汇总 Stage 6 Capability Probe")
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percent(value: float | None) -> str:
    """格式化可空百分比。"""
    return "N/A" if value is None else f"{value * 100:.2f}%"


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    """提取决定研究方向所需的最小指标集合。"""
    overall = summary["overall"]
    sample_types = summary["by_sample_type"]
    return {
        "direct_overall": overall["direct_compatibility_accuracy"],
        "explicit_overall": overall["explicit_compatibility_accuracy"],
        "near_miss_direct": sample_types["near_miss"]["direct_compatibility_accuracy"],
        "near_miss_explicit": sample_types["near_miss"]["explicit_compatibility_accuracy"],
        "valid_direct": sample_types["valid"]["direct_compatibility_accuracy"],
        "valid_explicit": sample_types["valid"]["explicit_compatibility_accuracy"],
        "functional_direct": sample_types["functional_mismatch"]["direct_compatibility_accuracy"],
        "decision_consistency": overall["decision_consistency"],
        "localization_exact": overall["violation_localization_accuracy"],
        "localization_field": overall["violation_field_accuracy"],
        "localization_constraint": overall["violation_constraint_accuracy"],
    }


def render_diagnosis(results: dict[str, dict[str, Any]]) -> str:
    """生成 Capability Diagnosis Report。"""
    lines = [
        "# Stage 6 Capability Diagnosis Report",
        "",
        "## 实验设置",
        "",
        "- 数据：30 个独立基础意图，每个意图包含 Valid、Near-miss、功能不匹配三个单工具样本，共 90 条。",
        "- 约束：enum、numeric range、训练未见的 string pattern，以及功能不匹配。",
        "- Probe A：原工具调用协议，直接调用或拒绝。",
        "- Probe B：显式 JSON 兼容性判断，同时要求定位违反约束。",
        "- 解码：固定贪心解码；五个模型使用完全相同的数据与提示。",
        "",
        "## 核心结果",
        "",
        "| 模型 | Direct Overall | Explicit Overall | Near-miss Direct | Near-miss Explicit | Valid Direct | Valid Explicit | Consistency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = results[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {percent(item['direct_overall'])} | "
            f"{percent(item['explicit_overall'])} | {percent(item['near_miss_direct'])} | "
            f"{percent(item['near_miss_explicit'])} | {percent(item['valid_direct'])} | "
            f"{percent(item['valid_explicit'])} | {percent(item['decision_consistency'])} |"
        )
    lines.extend(
        [
            "",
            "## 约束定位",
            "",
            "| 模型 | Exact | Field | Constraint Type |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = results[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {percent(item['localization_exact'])} | "
            f"{percent(item['localization_field'])} | {percent(item['localization_constraint'])} |"
        )
    lines.extend(
        [
            "",
            "## 诊断结论",
            "",
            "1. 功能不匹配不是主要瓶颈。除 Zero-shot 外，所有后训练模型在原协议下均达到 96.67%–100%。",
            "2. Near-miss 是关键断点。Zero-shot、LoRA、FDSRC 在原协议均为 0%，FCTW 仅 13.33%；"
            "但显式兼容性判断达到 96.67%–100%。模型能提取冲突信号，却没有稳定将其转化为拒绝行动。",
            "3. FWCT 将 Near-miss 原协议准确率提高到 46.67%，证明训练分布可以影响能力门控；"
            "其 Valid 准确率同时降至 70%，与已有 Distractor 错误拒绝现象一致。",
            "4. FCTW 将 Valid 恢复到 86.67%，却把 Near-miss 降到 13.33%。这说明四象限平衡没有形成"
            "稳定的能力判别边界，而是在调用和拒绝倾向之间重新折中。",
            "5. 显式协议对 Valid 存在明显过度拒绝，且字段/约束精确定位很弱。"
            "因此不能把高 Near-miss 二分类准确率解释为完整的 Schema 语义绑定能力。",
            "6. 当前主要瓶颈是 Capability-Aware Decision Integration；其次是可验证的字段级绑定。"
            "参数生成仍影响端到端结果，但不能解释 Substitution 的系统性失败。",
            "",
            "## 证据边界",
            "",
            "- Probe 使用历史诊断来源，不作为新方法的独立最终泛化证据。",
            "- 显式提示改变了任务形式，其结果只能证明信号可被提示激活，不能证明模型形成了稳定内部模块。",
            "- 约束定位采用严格字符串标签；Field 与 Constraint Type 另行报告，未使用模糊匹配美化结果。",
            "",
        ]
    )
    return "\n".join(lines)


def render_selection() -> str:
    """生成基于诊断证据的方法选择报告。"""
    return """# Stage 6 Method Selection Report

## 选择

本阶段选择 **Schema-Grounded Capability Integration**，采用“模型语义绑定 + 确定性 Schema 校验”的最小混合方案。

第一步已经实现 Schema Symbolic Verification：对模型提出的每个调用执行确定性 Schema 校验，只删除未知工具或参数违反当前 Schema 的调用，不猜测、不补全、不修复参数。

动态工具世界的实测结果表明，该操作可以直接修复大量 Capability Substitution 错误，同时不改变 Original 与 Distractor 的端到端准确率。因此，当前没有证据支持立即增加一次 SFT；下一步应先解决校验器无法观察到的 Requirement-to-Schema Binding。

## 为什么不先做 Capability Certificate SFT

- 五个模型在显式二分类下已经能识别大多数 Near-miss，新增二分类监督不直接针对主要断点。
- 约束精确定位接近零，直接训练完整证书会同时混入格式学习、字段绑定和行动学习，难以解释收益来源。
- 显式提示对有效工具存在过度拒绝，若直接把该输出作为训练目标，可能重复 FWCT 的拒绝偏置。

## 为什么保留 Symbolic Baseline

- Enum、Range、Pattern 等形式化约束应由确定性程序可靠验证，不应默认要求 1.5B 模型学习规则执行。
- 但只校验模型已生成的参数无法发现“用户明确要求 Patent，模型却省略可选 case_type”这类需求遗漏。
- 因此 Symbolic Baseline 既是强对照，也能明确剩余问题是否属于 Requirement-to-Schema Binding。

## 已验证的作用与边界

- FDSRC、FWCT、FCTW 的动态工具世界 Decision Exact 分别提高 14.66、5.67、13.67 个百分点。
- Capability Substitution 分别从 0.00%、56.67%、3.33% 提高到 66.67%、80.00%、66.67%。
- 三个模型均未出现原本端到端正确、过滤后错误的样本。
- 在 70 条 BFCL holdout 上，Arg Exact 保持不变，但 Tool Accuracy 因错误类型迁移下降 2.86 至 4.29 个百分点；硬过滤不能被宣称为普通工具调用的无损替代。
- 对用户要求映射到可选字段、同名字段语义绑定以及多调用部分完成等问题，单纯 Schema 校验无法处理。

## 最小后续实验

冻结一组新的独立工具世界，只实现“用户需求抽取到 Schema 字段”的轻量语义绑定层，再由相同的确定性校验器验证。只有该方法仍不能解决剩余错误时，才构造少量 Decision-Linked SFT；不增加证书长度、复杂损失或额外分类头。
"""


def main() -> None:
    """读取五个 Probe 汇总并写入两份研究报告。"""
    args = parse_args()
    probe_root = args.probe_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        path = probe_root / f"{method}_evaluation" / f"{method}_summary.json"
        results[method] = compact(json.loads(path.read_text(encoding="utf-8")))

    comparison_path = output_dir / "capability_probe_comparison.json"
    diagnosis_path = output_dir / "capability_diagnosis_report.md"
    selection_path = output_dir / "method_selection_report.md"
    for path in (comparison_path, diagnosis_path, selection_path):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    comparison_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnosis_path.write_text(render_diagnosis(results), encoding="utf-8")
    selection_path.write_text(render_selection(), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
