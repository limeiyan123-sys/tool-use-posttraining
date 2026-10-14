"""FCTW 四象限数据构造测试。"""

from __future__ import annotations

import unittest

from evaluation.fctw_metrics import aggregate_fctw, evaluate_fctw_case
from scripts.build_fctw_sft import build_factorial_worlds, find_constraint


def sample_base() -> dict:
    """构造同时支持 enum 与数值范围约束的单目标样本。"""
    return {
        "source_id": "simple_python_fctw_test",
        "category": "simple_python",
        "messages": [{"role": "user", "content": "Book 3 seats in Wuhan."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "book_seats",
                    "description": "Book seats in a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["city", "count"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_lookup",
                    "description": "Look up weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            },
        ],
        "expected_calls": [
            {
                "name": "book_seats",
                "arguments": {"city": ["Wuhan"], "count": [3]},
            }
        ],
    }


class FactorialConstructionTest(unittest.TestCase):
    """验证四象限标签、预算和约束语义。"""

    def test_enum_factorial_worlds_are_balanced(self) -> None:
        """enum 约束的四象限应具有相同工具数和 2:2 调用比例。"""
        base = sample_base()
        constraint = find_constraint(base, "enum", 7)
        self.assertIsNotNone(constraint)
        worlds = build_factorial_worlds(base, constraint, 7)
        self.assertEqual({world["world_type"] for world in worlds}, {"E10", "E01", "E11", "E00"})
        self.assertEqual(len({len(world["tools"]) for world in worlds}), 1)
        self.assertEqual(sum(bool(world["expected_calls"]) for world in worlds), 2)
        self.assertEqual(sum(not world["expected_calls"] for world in worlds), 2)

    def test_numeric_range_excludes_gold_value(self) -> None:
        """数值上界必须严格排除标准调用中的数值。"""
        base = sample_base()
        constraint = find_constraint(base, "numeric_range", 11)
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint["parameter_name"], "count")
        self.assertLess(constraint["schema_update"]["maximum"], 3)
        worlds = build_factorial_worlds(base, constraint, 11)
        near_world = next(world for world in worlds if world["world_type"] == "E01")
        near_name = near_world["transformation"]["near_miss_tool"]
        near_tool = next(
            tool for tool in near_world["tools"] if tool["function"]["name"] == near_name
        )
        self.assertLess(
            near_tool["function"]["parameters"]["properties"]["count"]["maximum"], 3
        )

    def test_multiple_target_tools_are_rejected(self) -> None:
        """存在多个必要工具时不得生成含糊的 E01/E00 Reject 标签。"""
        base = sample_base()
        base["expected_calls"].append(
            {"name": "weather_lookup", "arguments": {"city": ["Wuhan"]}}
        )
        self.assertIsNone(find_constraint(base, "enum", 13))

    def test_string_pattern_is_reserved_for_unseen_evaluation(self) -> None:
        """字符串 pattern 应能构造并排除训练样本中的 Gold 字符串。"""
        base = sample_base()
        constraint = find_constraint(base, "string_pattern", 17)
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint["parameter_name"], "city")
        self.assertIn("pattern", constraint["schema_update"])
        worlds = build_factorial_worlds(base, constraint, 17, track="test", id_prefix="test")
        self.assertEqual({world["track"] for world in worlds}, {"test"})


class FactorialMetricTest(unittest.TestCase):
    """验证 FCTW 错误归因与分组指标。"""

    def test_false_rejection_is_counted_in_e11(self) -> None:
        """E11 存在有效工具却无调用时必须归因于错误拒绝。"""
        base = sample_base()
        constraint = find_constraint(base, "enum", 19)
        worlds = build_factorial_worlds(base, constraint, 19)
        results = []
        for world in worlds:
            if world["world_type"] == "E11":
                response = "The current tool environment cannot complete this request."
            elif world["expected_calls"]:
                call = world["expected_calls"][0]
                response = (
                    "<tool_call>"
                    + __import__("json").dumps(
                        {"name": call["name"], "arguments": {"city": "Wuhan", "count": 3}}
                    )
                    + "</tool_call>"
                )
            else:
                response = "No compatible tool."
            results.append(evaluate_fctw_case(world, {"response": response}))
        summary = aggregate_fctw(results)
        self.assertEqual(summary["by_world"]["E11"]["false_rejection_rate"], 1.0)
        self.assertEqual(summary["by_world"]["E10"]["decision_exact"], 1.0)


if __name__ == "__main__":
    unittest.main()
