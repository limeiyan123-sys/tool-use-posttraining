"""验证 CCPD-v1.1 F1 严格响应分类与调用匹配。"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from evaluation.metrics import parse_tool_calls


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ccpd_v11_know_act"
    / "f1_prompt"
    / "analyze_f1.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_f1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class F1AnalysisTests(unittest.TestCase):
    """覆盖拒绝语义、直接回答和参数精确匹配。"""

    def test_explicit_rejection_is_decline(self) -> None:
        """明确拒绝文本应识别为合法 DECLINE。"""
        decision, valid, calls = MODULE.classify_response(
            "无法使用当前工具完成请求。<|im_end|>", parse_tool_calls
        )
        self.assertEqual((decision, valid, calls), ("DECLINE", True, []))

    def test_direct_answer_is_not_silent_decline(self) -> None:
        """无工具调用的直接答案不能回退为正确拒绝。"""
        decision, valid, calls = MODULE.classify_response("The answer is 42.", parse_tool_calls)
        self.assertEqual((decision, valid, calls), ("INVALID", False, []))

    def test_tool_call_exact_requires_all_arguments(self) -> None:
        """完整调用必须同时匹配工具名、参数集合和值。"""
        expected = [{"name": "weather", "arguments": {"city": "Paris", "date": "2026-10-17"}}]
        correct = [{"name": "weather", "arguments": {"city": "Paris", "date": "2026-10-17"}}]
        missing = [{"name": "weather", "arguments": {"city": "Paris"}}]
        self.assertTrue(MODULE.call_exact(correct, expected))
        self.assertFalse(MODULE.call_exact(missing, expected))


if __name__ == "__main__":
    unittest.main()
