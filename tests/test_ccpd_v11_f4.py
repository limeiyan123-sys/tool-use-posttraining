"""CCPD-v1.1 F4 数据转换测试。"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "ccpd_v11_know_act" / "f4_sft" / "build_f4_dataset.py"
SPEC = importlib.util.spec_from_file_location("ccpd_v11_f4_build", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class F4DatasetTest(unittest.TestCase):
    """验证调用与拒绝监督严格共享同一基础请求。"""

    def setUp(self) -> None:
        """构造一条最小、明确的闭世界配对。"""
        tool = {
            "type": "function",
            "function": {
                "name": "book",
                "description": "预订",
                "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}},
            },
        }
        self.pair = {
            "pair_id": "p1",
            "split": "train",
            "constraint_family": "numeric_maximum",
            "tool_family": "book",
            "plus": {
                "messages": [{"role": "user", "content": "预订三天"}],
                "tools": [tool],
                "expected_calls": [{"name": "book", "arguments": {"days": 3}}],
            },
            "minus": {
                "messages": [{"role": "user", "content": "预订三天"}],
                "tools": [tool],
                "expected_calls": [],
            },
        }

    def test_balanced_targets(self) -> None:
        """可满足侧是函数调用，不可满足侧是固定拒绝。"""
        plus = MODULE.make_example(self.pair, "plus")
        minus = MODULE.make_example(self.pair, "minus")
        self.assertEqual(plus["conversations"][1]["from"], "function_call")
        self.assertEqual(minus["conversations"][1]["from"], "gpt")
        self.assertEqual(minus["conversations"][1]["value"], MODULE.REJECTION)
        self.assertEqual(plus["conversations"][0], minus["conversations"][0])

    def test_tools_are_normalized(self) -> None:
        """LLaMA-Factory 工具字段不保留外层 type/function wrapper。"""
        example = MODULE.make_example(self.pair, "plus")
        tools = json.loads(example["tools"])
        self.assertEqual(tools[0]["name"], "book")
        self.assertNotIn("function", tools[0])


if __name__ == "__main__":
    unittest.main()
