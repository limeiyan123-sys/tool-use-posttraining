"""验证 CCPD-v1.1 F0 合同构造的关键不变量。"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ccpd_v11_know_act"
    / "f0_data"
    / "build_f0_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("build_f0_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class F0ContractTests(unittest.TestCase):
    """覆盖约束方向、执行语义和模板规范化。"""

    def test_numeric_constraints_accept_plus_and_reject_minus(self) -> None:
        """数值上下界两种方向都必须满足 E+/E- 定义。"""
        for salt in ("a", "b", "c", "d"):
            plus, minus, family = MODULE.build_constraints(30, salt, "seen")
            self.assertIn(family, {"numeric_minimum", "numeric_maximum"})
            field = {"type": "integer", "description": "value", **plus}
            tool = {
                "type": "function",
                "function": {
                    "name": "tool_x",
                    "description": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": field},
                        "required": ["value"],
                    },
                },
            }
            self.assertTrue(MODULE.schema_accepts({"value": 30}, tool))
            tool["function"]["parameters"]["properties"]["value"] = {
                "type": "integer",
                **minus,
            }
            self.assertFalse(MODULE.schema_accepts({"value": 30}, tool))

    def test_enum_constraints_have_equal_cardinality(self) -> None:
        """字符串两侧必须采用相同枚举基数，避免数量捷径。"""
        plus, minus, family = MODULE.build_constraints("suite", "hotel", "seen")
        self.assertEqual(family, "enum")
        self.assertEqual(len(plus["enum"]), len(minus["enum"]))
        self.assertIn("suite", plus["enum"])
        self.assertNotIn("suite", minus["enum"])

    def test_const_is_reserved_unseen_family(self) -> None:
        """未见约束划分使用同类 const 约束且两侧执行结果相反。"""
        plus, minus, family = MODULE.build_constraints("express", "shipping", "unseen_const")
        self.assertEqual(family, "const")
        self.assertEqual(plus["const"], "express")
        self.assertNotEqual(minus["const"], "express")

    def test_remove_constraint_restores_descriptions(self) -> None:
        """约束类型改写前必须同时清理 Schema 键和自然语言约束。"""
        neutral = {
            "type": "function",
            "function": {
                "name": "tool_x",
                "description": "Book a room.",
                "parameters": {
                    "type": "object",
                    "properties": {"room": {"type": "string", "description": "Room type."}},
                    "required": ["room"],
                },
            },
        }
        constraint = {"enum": ["suite", "deluxe"]}
        constrained = MODULE.apply_constraint(neutral, "tool_x", "room", constraint)
        restored = MODULE.remove_constraint(constrained, "room", constraint)
        self.assertEqual(restored, neutral)

    def test_query_template_removes_target_and_numbers(self) -> None:
        """请求模板不应因实体值或普通数值差异跨划分泄漏。"""
        left = MODULE.query_template("Book a suite for 3 nights", "suite")
        right = MODULE.query_template("Book a deluxe room for 8 nights", "deluxe room")
        self.assertEqual(left, right)

    def test_normalize_schema_preserves_non_string_type(self) -> None:
        """BFCL 中的复杂 type 对象必须保留，不能被错误映射或触发异常。"""
        schema = {"type": {"kind": "union"}, "properties": {"x": {"type": "float"}}}
        normalized = MODULE.normalize_schema(schema)
        self.assertEqual(normalized["type"], {"kind": "union"})
        self.assertEqual(normalized["properties"]["x"]["type"], "number")


if __name__ == "__main__":
    unittest.main()
