"""CCPD E1/E2 数据审计与奖励预检的回归测试。"""

from __future__ import annotations

import math
import unittest

from experiments.ccpd_core_validation.e1_diagnosis.audit_pairs import (
    check_constraint_excludes,
)
from experiments.ccpd_core_validation.e2_pair_pilot.build_e2_pilot import (
    has_strict_upper_bound_ambiguity,
    positive_constraint,
    schema_accepts,
)
from experiments.ccpd_core_validation.e2_pair_pilot.preflight_rollouts import (
    standardized_advantages,
)


class CcpdE1E2Test(unittest.TestCase):
    """验证 E1/E2 最关键的 Gold 与组内优势不变量。"""

    def test_negative_constraint_excludes_gold(self) -> None:
        """枚举和数值上界必须真实排除 Gold 值。"""
        self.assertTrue(check_constraint_excludes("suite", {"enum": ["standard"]}))
        self.assertFalse(check_constraint_excludes("suite", {"enum": ["suite"]}))
        self.assertTrue(check_constraint_excludes(10, {"maximum": 9}))
        self.assertFalse(check_constraint_excludes(10, {"maximum": 10}))

    def test_positive_constraint_contains_all_gold_values(self) -> None:
        """中性合同必须包含并行调用涉及的全部目标值。"""
        self.assertEqual(positive_constraint("enum", ["A", "B", "A"], {"enum": ["X"]}), {"enum": ["A", "B"]})
        self.assertEqual(positive_constraint("numeric_range", [5, 10], {"maximum": 4}), {"maximum": 10})

    def test_strict_upper_bound_wording_is_rejected(self) -> None:
        """自然语言严格小于不能被 maximum 的闭区间错误接纳。"""
        self.assertTrue(has_strict_upper_bound_ambiguity("finish under 30 minutes", [30], "numeric_range"))
        self.assertTrue(has_strict_upper_bound_ambiguity("use less than 5 items", [5], "numeric_range"))
        self.assertFalse(has_strict_upper_bound_ambiguity("within 30 minutes", [30], "numeric_range"))

    def test_schema_subset_distinguishes_pair(self) -> None:
        """同一个 Gold 调用应通过 E+ 并被 E- 拒绝。"""
        template = {
            "function": {
                "parameters": {
                    "type": "object",
                    "required": ["room_type"],
                    "properties": {"room_type": {"type": "string"}},
                }
            }
        }
        plus = {"function": {"parameters": {**template["function"]["parameters"], "properties": {"room_type": {"type": "string", "enum": ["suite"]}}}}}
        minus = {"function": {"parameters": {**template["function"]["parameters"], "properties": {"room_type": {"type": "string", "enum": ["standard"]}}}}}
        self.assertTrue(schema_accepts({"room_type": "suite"}, plus))
        self.assertFalse(schema_accepts({"room_type": "suite"}, minus))

    def test_zero_variance_rewards_produce_no_advantage(self) -> None:
        """全零或全一联合奖励组不能产生 GRPO 优势。"""
        self.assertEqual(standardized_advantages([0.0] * 8), [0.0] * 8)
        self.assertEqual(standardized_advantages([1.0] * 8), [0.0] * 8)
        advantages = standardized_advantages([1.0, 0.0])
        self.assertTrue(math.isclose(advantages[0], 1.0))
        self.assertTrue(math.isclose(advantages[1], -1.0))


if __name__ == "__main__":
    unittest.main()
