"""CCPD E0：解析梯度、自动微分与 GRPO 分组期望更新检查。"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
ROLLOUTS_PER_SIDE = 4
EPSILON = 1e-12


@dataclass(frozen=True)
class Scenario:
    """定义一组可满足侧与不可满足侧的调用概率。"""

    name: str
    p_plus: float
    p_minus: float
    interpretation: str


SCENARIOS = [
    Scenario("高调用倾向", 0.90, 0.90, "E+ 正确、E- 错误"),
    Scenario("高拒绝倾向", 0.10, 0.10, "E+ 错误、E- 正确"),
    Scenario("两侧接近", 0.50, 0.50, "尚无合同区分"),
    Scenario("强区分", 0.90, 0.10, "两侧均大概率正确"),
    Scenario("反向区分", 0.10, 0.90, "两侧均大概率错误"),
    Scenario("中等偏调用", 0.75, 0.45, "E+ 较好、E- 仍常误调用"),
]


def sigmoid(value: float) -> float:
    """计算标量 Logistic 函数。"""
    return 1.0 / (1.0 + math.exp(-value))


def logit(probability: float) -> float:
    """将开区间概率转换为对数几率。"""
    if not 0.0 < probability < 1.0:
        raise ValueError("概率必须位于 (0, 1)")
    return math.log(probability / (1.0 - probability))


def analytical_metrics(p_plus: float, p_minus: float) -> dict[str, float]:
    """返回两个目标及其对 b、d+、d- 和合同差异 delta 的解析梯度。"""
    a_plus = p_plus * (1.0 - p_plus)
    a_minus = p_minus * (1.0 - p_minus)
    balanced = 0.5 * (p_plus + 1.0 - p_minus)
    pair = p_plus * (1.0 - p_minus)
    balanced_d_plus = 0.5 * a_plus
    balanced_d_minus = -0.5 * a_minus
    pair_d_plus = pair * (1.0 - p_plus)
    pair_d_minus = -pair * p_minus
    return {
        "j_balanced": balanced,
        "j_pair": pair,
        "balanced_grad_b": balanced_d_plus + balanced_d_minus,
        "pair_grad_b": pair_d_plus + pair_d_minus,
        "balanced_grad_d_plus": balanced_d_plus,
        "balanced_grad_d_minus": balanced_d_minus,
        "pair_grad_d_plus": pair_d_plus,
        "pair_grad_d_minus": pair_d_minus,
        # 固定合同均值，令 d+=m+delta/2、d-=m-delta/2。
        "balanced_grad_delta": 0.5 * (balanced_d_plus - balanced_d_minus),
        "pair_grad_delta": 0.5 * (pair_d_plus - pair_d_minus),
        "pair_to_balanced_plus_ratio": 2.0 * (1.0 - p_minus),
        "pair_to_balanced_minus_ratio": 2.0 * p_plus,
        "pair_zero_variance_probability": (1.0 - pair) ** ROLLOUTS_PER_SIDE
        + pair**ROLLOUTS_PER_SIDE,
    }


def autograd_metrics(p_plus: float, p_minus: float) -> dict[str, float]:
    """使用 PyTorch 自动微分复核解析梯度。"""
    z_plus = torch.tensor(logit(p_plus), dtype=torch.float64, requires_grad=True)
    z_minus = torch.tensor(logit(p_minus), dtype=torch.float64, requires_grad=True)
    plus = torch.sigmoid(z_plus)
    minus = torch.sigmoid(z_minus)
    balanced = 0.5 * (plus + 1.0 - minus)
    pair = plus * (1.0 - minus)
    balanced_grads = torch.autograd.grad(balanced, (z_plus, z_minus), retain_graph=True)
    pair_grads = torch.autograd.grad(pair, (z_plus, z_minus))
    return {
        "balanced_grad_d_plus": balanced_grads[0].item(),
        "balanced_grad_d_minus": balanced_grads[1].item(),
        "pair_grad_d_plus": pair_grads[0].item(),
        "pair_grad_d_minus": pair_grads[1].item(),
    }


def normalized_advantages(rewards: np.ndarray) -> np.ndarray:
    """按组执行总体标准差标准化；零方差组返回全零优势。"""
    centered = rewards - rewards.mean()
    standard_deviation = math.sqrt(float(np.mean(centered**2)))
    if standard_deviation <= EPSILON:
        return np.zeros_like(rewards, dtype=np.float64)
    return centered / standard_deviation


def outcome_probability(actions: tuple[int, ...], probabilities: tuple[float, ...]) -> float:
    """计算独立伯努利动作序列的联合概率。"""
    result = 1.0
    for action, probability in zip(actions, probabilities):
        result *= probability if action else 1.0 - probability
    return result


def exact_grpo_expectation(p_plus: float, p_minus: float, mode: str) -> dict[str, float]:
    """枚举 8 条 rollout，计算三种分组方式的期望标准化策略梯度。"""
    if mode not in {"separate_independent", "mixed_independent", "mixed_paired"}:
        raise ValueError(f"未知模式：{mode}")
    probabilities = (p_plus,) * ROLLOUTS_PER_SIDE + (p_minus,) * ROLLOUTS_PER_SIDE
    expected_plus = 0.0
    expected_minus = 0.0
    nonzero_group_probability = 0.0
    expected_nonzero_advantages = 0.0
    reward_mean = 0.0
    reward_variance = 0.0
    all_zero_reward_probability = 0.0
    all_one_reward_probability = 0.0
    for actions in itertools.product((0, 1), repeat=2 * ROLLOUTS_PER_SIDE):
        probability = outcome_probability(actions, probabilities)
        plus_actions = np.asarray(actions[:ROLLOUTS_PER_SIDE], dtype=np.float64)
        minus_actions = np.asarray(actions[ROLLOUTS_PER_SIDE:], dtype=np.float64)
        if mode == "mixed_paired":
            pair_rewards = plus_actions * (1.0 - minus_actions)
            rewards = np.concatenate([pair_rewards, pair_rewards])
            advantages = normalized_advantages(rewards)
            group_nonzero = bool(np.any(np.abs(advantages) > EPSILON))
        else:
            plus_rewards = plus_actions
            minus_rewards = 1.0 - minus_actions
            rewards = np.concatenate([plus_rewards, minus_rewards])
            if mode == "mixed_independent":
                advantages = normalized_advantages(rewards)
                group_nonzero = bool(np.any(np.abs(advantages) > EPSILON))
            else:
                plus_advantages = normalized_advantages(plus_rewards)
                minus_advantages = normalized_advantages(minus_rewards)
                advantages = np.concatenate([plus_advantages, minus_advantages])
                group_nonzero = bool(
                    np.any(np.abs(plus_advantages) > EPSILON)
                    or np.any(np.abs(minus_advantages) > EPSILON)
                )
        plus_scores = plus_actions - p_plus
        minus_scores = minus_actions - p_minus
        expected_plus += probability * float(np.mean(advantages[:ROLLOUTS_PER_SIDE] * plus_scores)) / 2.0
        expected_minus += probability * float(np.mean(advantages[ROLLOUTS_PER_SIDE:] * minus_scores)) / 2.0
        nonzero_group_probability += probability * float(group_nonzero)
        expected_nonzero_advantages += probability * float(np.mean(np.abs(advantages) > EPSILON))
        reward_mean += probability * float(rewards.mean())
        reward_variance += probability * float(np.var(rewards))
        all_zero_reward_probability += probability * float(np.all(rewards == 0.0))
        all_one_reward_probability += probability * float(np.all(rewards == 1.0))
    return {
        "expected_grad_z_plus": expected_plus,
        "expected_grad_z_minus": expected_minus,
        "expected_grad_common": expected_plus + expected_minus,
        "expected_grad_contrast": 0.5 * (expected_plus - expected_minus),
        "nonzero_group_probability": nonzero_group_probability,
        "expected_nonzero_advantage_fraction": expected_nonzero_advantages,
        "expected_reward_mean": reward_mean,
        "expected_within_group_reward_variance": reward_variance,
        "all_zero_reward_probability": all_zero_reward_probability,
        "all_one_reward_probability": all_one_reward_probability,
    }


def unnormalized_policy_gradient(p_plus: float, p_minus: float, paired: bool) -> dict[str, float]:
    """计算无组标准化时，平均每条 rollout 的解析策略梯度。"""
    if paired:
        pair = p_plus * (1.0 - p_minus)
        plus = 0.5 * pair * (1.0 - p_plus)
        minus = -0.5 * pair * p_minus
    else:
        plus = 0.5 * p_plus * (1.0 - p_plus)
        minus = -0.5 * p_minus * (1.0 - p_minus)
    return {
        "expected_grad_z_plus": plus,
        "expected_grad_z_minus": minus,
        "expected_grad_common": plus + minus,
        "expected_grad_contrast": 0.5 * (plus - minus),
    }


def cost_sensitive_check(p_plus: float, p_minus: float, alpha: float) -> dict[str, float]:
    """比较 J_alpha 真梯度与联合成功侧权重估计器的期望方向。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须位于 (0, 1)")
    objective = p_plus**alpha * (1.0 - p_minus) ** (1.0 - alpha)
    pair = p_plus * (1.0 - p_minus)
    true_plus = objective * alpha * (1.0 - p_plus)
    true_minus = -objective * (1.0 - alpha) * p_minus
    estimator_plus = pair * alpha * (1.0 - p_plus)
    estimator_minus = -pair * (1.0 - alpha) * p_minus
    return {
        "alpha": alpha,
        "j_alpha": objective,
        "true_grad_z_plus": true_plus,
        "true_grad_z_minus": true_minus,
        "joint_weighted_expected_z_plus": estimator_plus,
        "joint_weighted_expected_z_minus": estimator_minus,
        "common_positive_scale": pair / objective,
    }


def exact_cost_sensitive_grpo(p_plus: float, p_minus: float, alpha: float) -> dict[str, float]:
    """枚举侧权重联合奖励在混合组标准化后的期望更新。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须位于 (0, 1)")
    probabilities = (p_plus,) * ROLLOUTS_PER_SIDE + (p_minus,) * ROLLOUTS_PER_SIDE
    expected_plus = 0.0
    expected_minus = 0.0
    nonzero_group_probability = 0.0
    for actions in itertools.product((0, 1), repeat=2 * ROLLOUTS_PER_SIDE):
        probability = outcome_probability(actions, probabilities)
        plus_actions = np.asarray(actions[:ROLLOUTS_PER_SIDE], dtype=np.float64)
        minus_actions = np.asarray(actions[ROLLOUTS_PER_SIDE:], dtype=np.float64)
        pair_rewards = plus_actions * (1.0 - minus_actions)
        rewards = np.concatenate([alpha * pair_rewards, (1.0 - alpha) * pair_rewards])
        advantages = normalized_advantages(rewards)
        plus_scores = plus_actions - p_plus
        minus_scores = minus_actions - p_minus
        expected_plus += probability * float(np.mean(advantages[:ROLLOUTS_PER_SIDE] * plus_scores)) / 2.0
        expected_minus += probability * float(np.mean(advantages[ROLLOUTS_PER_SIDE:] * minus_scores)) / 2.0
        nonzero_group_probability += probability * float(np.any(np.abs(advantages) > EPSILON))
    theoretical = cost_sensitive_check(p_plus, p_minus, alpha)
    true_vector = np.asarray([theoretical["true_grad_z_plus"], theoretical["true_grad_z_minus"]])
    grpo_vector = np.asarray([expected_plus, expected_minus])
    denominator = float(np.linalg.norm(true_vector) * np.linalg.norm(grpo_vector))
    cosine = float(np.dot(true_vector, grpo_vector) / denominator) if denominator > EPSILON else float("nan")
    return {
        "standardized_expected_z_plus": expected_plus,
        "standardized_expected_z_minus": expected_minus,
        "standardized_common_gradient": expected_plus + expected_minus,
        "standardized_contrast_gradient": 0.5 * (expected_plus - expected_minus),
        "standardized_to_true_cosine": cosine,
        "standardized_nonzero_group_probability": nonzero_group_probability,
    }


def verify_autograd(rows: list[dict[str, float | str]]) -> float:
    """返回解析梯度与自动微分之间的最大绝对误差。"""
    maximum_error = 0.0
    for row in rows:
        automatic = autograd_metrics(float(row["p_plus"]), float(row["p_minus"]))
        for key, value in automatic.items():
            maximum_error = max(maximum_error, abs(float(row[key]) - value))
    if maximum_error > 1e-12:
        raise AssertionError(f"自动微分检查失败，最大误差 {maximum_error}")
    return maximum_error


def write_csv(path: Path, rows: list[dict]) -> None:
    """将同构记录保存为 UTF-8 CSV。"""
    if not rows:
        raise ValueError("不能写入空表")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures() -> None:
    """生成目标梯度与奖励稀疏性的两张简洁图。"""
    probabilities = np.linspace(0.02, 0.98, 97)
    pair_zero_variance = np.zeros((len(probabilities), len(probabilities)))
    contrast_ratio = np.zeros_like(pair_zero_variance)
    for row_index, p_minus in enumerate(probabilities):
        for column_index, p_plus in enumerate(probabilities):
            metrics = analytical_metrics(float(p_plus), float(p_minus))
            pair_zero_variance[row_index, column_index] = metrics["pair_zero_variance_probability"]
            denominator = metrics["balanced_grad_delta"]
            contrast_ratio[row_index, column_index] = metrics["pair_grad_delta"] / denominator

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    first = axes[0].imshow(
        contrast_ratio,
        origin="lower",
        extent=[0.02, 0.98, 0.02, 0.98],
        aspect="auto",
        cmap="coolwarm",
        vmin=0,
        vmax=2,
    )
    axes[0].set_title("Pair / balanced contrast-gradient ratio")
    axes[0].set_xlabel("p_plus")
    axes[0].set_ylabel("p_minus")
    figure.colorbar(first, ax=axes[0])
    second = axes[1].imshow(
        pair_zero_variance,
        origin="lower",
        extent=[0.02, 0.98, 0.02, 0.98],
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Paired reward: zero-variance group probability")
    axes[1].set_xlabel("p_plus")
    axes[1].set_ylabel("p_minus")
    figure.colorbar(second, ax=axes[1])
    figure.savefig(OUTPUT_DIR / "gradient_and_sparsity.png", dpi=180)
    plt.close(figure)

    labels = ["High call", "High decline", "Near 0.5", "Strong split", "Reversed split", "Moderate call"]
    x_positions = np.arange(len(labels))
    modes = [
        ("separate_independent", "Independent / separate"),
        ("mixed_independent", "Independent / mixed"),
        ("mixed_paired", "Paired reward / mixed"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    width = 0.25
    for mode_index, (mode, display_name) in enumerate(modes):
        statistics = [exact_grpo_expectation(item.p_plus, item.p_minus, mode) for item in SCENARIOS]
        axes[0].bar(
            x_positions + (mode_index - 1) * width,
            [item["expected_grad_contrast"] for item in statistics],
            width,
            label=display_name,
        )
        axes[1].bar(
            x_positions + (mode_index - 1) * width,
            [item["nonzero_group_probability"] for item in statistics],
            width,
            label=display_name,
        )
    for axis in axes:
        axis.set_xticks(x_positions, labels, rotation=24, ha="right")
        axis.axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Expected contract-contrast update")
    axes[0].set_ylabel("Expected gradient")
    axes[1].set_title("Probability of a non-zero-advantage group")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    figure.savefig(OUTPUT_DIR / "grpo_grouping_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """执行 E0 全部离线检查并写出结构化结果。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    objective_rows: list[dict] = []
    grpo_rows: list[dict] = []
    for scenario in SCENARIOS:
        metrics = analytical_metrics(scenario.p_plus, scenario.p_minus)
        objective_rows.append({**asdict(scenario), **metrics})
        for mode in ["separate_independent", "mixed_independent", "mixed_paired"]:
            grpo_rows.append(
                {
                    **asdict(scenario),
                    "mode": mode,
                    **exact_grpo_expectation(scenario.p_plus, scenario.p_minus, mode),
                }
            )
    maximum_autograd_error = verify_autograd(objective_rows)
    write_csv(OUTPUT_DIR / "objective_gradient_scan.csv", objective_rows)
    write_csv(OUTPUT_DIR / "grpo_grouping_scan.csv", grpo_rows)

    cost_sensitive_rows = [
        {
            **asdict(scenario),
            **cost_sensitive_check(scenario.p_plus, scenario.p_minus, alpha),
            **exact_cost_sensitive_grpo(scenario.p_plus, scenario.p_minus, alpha),
        }
        for scenario in SCENARIOS
        for alpha in (0.3, 0.5, 0.7)
    ]
    write_csv(OUTPUT_DIR / "cost_sensitive_scan.csv", cost_sensitive_rows)
    make_figures()

    summary = {
        "experiment": "E0",
        "rollouts_per_side": ROLLOUTS_PER_SIDE,
        "autograd_max_absolute_error": maximum_autograd_error,
        "scenarios": objective_rows,
        "grpo_grouping": grpo_rows,
        "cost_sensitive": cost_sensitive_rows,
        "conventions": {
            "gradient_direction": "目标最大化方向",
            "delta": "d_plus - d_minus，固定合同均值；报告梯度包含 1/2 链式系数",
            "group_standardization": "总体标准差；零方差组优势置零；不含 PPO clip、KL 与长度归一化",
            "rollout_budget": "每侧 4 条，总计 8 条；separate 模式在两个 4 条子组内分别标准化",
        },
    }
    (OUTPUT_DIR / "e0_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "autograd_max_absolute_error": maximum_autograd_error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
