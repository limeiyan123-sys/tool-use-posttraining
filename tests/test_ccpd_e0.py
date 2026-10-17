"""CCPD E0 理论脚本的回归测试。"""

from __future__ import annotations

import math
import unittest

from experiments.ccpd_core_validation.e0_theory.ccpd_e0_analysis import (
    analytical_metrics,
    autograd_metrics,
    cost_sensitive_check,
    exact_cost_sensitive_grpo,
    exact_grpo_expectation,
)


class CcpdE0Test(unittest.TestCase):
    """验证 E0 的关键数学不变量。"""

    def test_analytical_gradient_matches_autograd(self) -> None:
        """解析目标梯度必须与自动微分一致。"""
        for p_plus, p_minus in [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1), (0.75, 0.45)]:
            analytical = analytical_metrics(p_plus, p_minus)
            automatic = autograd_metrics(p_plus, p_minus)
            for key, value in automatic.items():
                self.assertTrue(math.isclose(analytical[key], value, rel_tol=0.0, abs_tol=1e-12))

    def test_balanced_and_pair_share_relevant_common_bias_stationary_point(self) -> None:
        """当两侧错误率相等时，两个目标对共同偏置的梯度均为零。"""
        for p_plus in (0.1, 0.25, 0.7, 0.9):
            metrics = analytical_metrics(p_plus, 1.0 - p_plus)
            self.assertTrue(math.isclose(metrics["balanced_grad_b"], 0.0, abs_tol=1e-12))
            self.assertTrue(math.isclose(metrics["pair_grad_b"], 0.0, abs_tol=1e-12))

    def test_pair_reward_is_sparse_when_both_sides_are_wrong(self) -> None:
        """反向区分状态下，四对联合奖励大概率形成零方差组。"""
        metrics = analytical_metrics(0.1, 0.9)
        self.assertGreater(metrics["pair_zero_variance_probability"], 0.96)
        paired = exact_grpo_expectation(0.1, 0.9, "mixed_paired")
        independent = exact_grpo_expectation(0.1, 0.9, "mixed_independent")
        self.assertLess(paired["nonzero_group_probability"], independent["nonzero_group_probability"])

    def test_joint_weighted_cost_estimator_has_same_unstandardized_direction(self) -> None:
        """无组标准化时，侧权重联合成功估计器与 J_alpha 梯度只差正标量。"""
        result = cost_sensitive_check(0.73, 0.41, 0.3)
        scale_plus = result["joint_weighted_expected_z_plus"] / result["true_grad_z_plus"]
        scale_minus = result["joint_weighted_expected_z_minus"] / result["true_grad_z_minus"]
        self.assertTrue(math.isclose(scale_plus, scale_minus, rel_tol=1e-12))
        self.assertTrue(math.isclose(scale_plus, result["common_positive_scale"], rel_tol=1e-12))

    def test_group_standardization_breaks_exact_cost_objective_correspondence(self) -> None:
        """混合组标准化后，侧权重实现通常不再与 J_alpha 梯度共线。"""
        result = exact_cost_sensitive_grpo(0.9, 0.1, 0.7)
        self.assertLess(result["standardized_to_true_cosine"], 0.999)


if __name__ == "__main__":
    unittest.main()
