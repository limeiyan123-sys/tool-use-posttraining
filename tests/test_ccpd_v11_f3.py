"""CCPD-v1.1 F3 判断协议的单元测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "ccpd_v11_know_act" / "f3_bridge" / "run_f3_bridge.py"
SPEC = importlib.util.spec_from_file_location("ccpd_v11_f3_run", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class F3ProtocolTest(unittest.TestCase):
    """验证严格判断解析和行动指令不会静默回退。"""

    def test_parse_decision_accepts_only_tagged_labels(self) -> None:
        """合法标签可解析，普通文本不得被猜测为判断。"""
        self.assertEqual(MODULE.parse_decision("<decision>CALL</decision>"), "CALL")
        self.assertEqual(MODULE.parse_decision("<decision> decline </decision>"), "DECLINE")
        self.assertEqual(MODULE.parse_decision("I think CALL"), "INVALID")

    def test_injected_guidance_is_explicit(self) -> None:
        """CALL 与 DECLINE 指令必须明确控制是否生成工具调用。"""
        self.assertIn("结构化工具调用", MODULE.guidance("CALL"))
        self.assertIn("不要生成工具调用", MODULE.guidance("DECLINE"))
        self.assertIn("UNKNOWN", MODULE.guidance("INVALID"))

    def test_expected_decision_rejects_unknown_side(self) -> None:
        """未知配对侧应抛出异常，不能使用隐式默认标签。"""
        self.assertEqual(MODULE.expected_decision("plus"), "CALL")
        self.assertEqual(MODULE.expected_decision("minus"), "DECLINE")
        with self.assertRaises(ValueError):
            MODULE.expected_decision("other")

    def test_text_decision_does_not_register_tools(self) -> None:
        """强基线应保留合同文本，同时把工具注册这一混杂因素移出判断阶段。"""
        side = {
            "messages": [{"role": "user", "content": "查询天气"}],
            "tools": [{"name": "weather", "parameters": {"type": "object"}}],
        }
        messages = MODULE.text_decision_messages(side)
        self.assertIn("查询天气", messages[1]["content"])
        self.assertIn("weather", messages[1]["content"])
        self.assertNotIn("tools", messages[1])


if __name__ == "__main__":
    unittest.main()
