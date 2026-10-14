"""工具调用评测器单元测试。"""

from __future__ import annotations

import unittest

from evaluation.metrics import argument_score, evaluate_case, parse_tool_calls, validate_arguments


class ParseToolCallsTest(unittest.TestCase):
    """验证严格解析策略。"""

    def test_valid_call(self) -> None:
        """合法调用应解析成功。"""
        result = parse_tool_calls('<tool_call>{"name":"calc","arguments":{"x":1}}</tool_call>')
        self.assertTrue(result.valid)
        self.assertEqual(result.calls[0]["name"], "calc")

    def test_unclosed_tag_is_error(self) -> None:
        """未闭合标签必须失败，不能静默修复。"""
        result = parse_tool_calls('<tool_call>{"name":"calc","arguments":{}}')
        self.assertFalse(result.valid)
        self.assertIn("未闭合", result.error or "")

    def test_extra_json_field_is_error(self) -> None:
        """调用对象出现额外顶层字段时必须失败。"""
        result = parse_tool_calls(
            '<tool_call>{"name":"calc","arguments":{},"thought":"hidden"}</tool_call>'
        )
        self.assertFalse(result.valid)

    def test_plain_answer_means_no_call(self) -> None:
        """普通文本应被解释为没有工具调用。"""
        result = parse_tool_calls("这是一个普通回答。")
        self.assertTrue(result.valid)
        self.assertEqual(result.calls, [])


class SchemaValidationTest(unittest.TestCase):
    """验证参数 Schema 检查。"""

    def test_bool_is_not_integer(self) -> None:
        """Python 中 bool 是 int 子类，但 JSON Schema 中必须区分。"""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        }
        errors = validate_arguments({"count": True}, schema)
        self.assertTrue(errors)

    def test_numeric_maximum_is_enforced(self) -> None:
        """数值超过 JSON Schema maximum 时必须报错。"""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer", "maximum": 2}},
            "required": ["count"],
        }
        self.assertTrue(validate_arguments({"count": 3}, schema))

    def test_string_pattern_is_enforced(self) -> None:
        """字符串不匹配 JSON Schema pattern 时必须报错。"""
        schema = {
            "type": "object",
            "properties": {"code": {"type": "string", "pattern": "^ok$"}},
            "required": ["code"],
        }
        self.assertTrue(validate_arguments({"code": "wrong"}, schema))

    def test_extra_argument_is_error(self) -> None:
        """关闭额外字段时，多余参数必须报错。"""
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        errors = validate_arguments({"city": "北京", "date": "2026-10-20"}, schema)
        self.assertIn("arguments.date 是未声明参数", errors)


class ArgumentScoreTest(unittest.TestCase):
    """验证 BFCL 候选值参数评分。"""

    def test_explicit_empty_string_is_allowed(self) -> None:
        """标准答案含空字符串时，显式空值和省略字段都应得分。"""
        explicit_score, explicit_exact = argument_score({"genre": ""}, {"genre": [""]})
        omitted_score, omitted_exact = argument_score({}, {"genre": [""]})
        self.assertEqual(explicit_score, 1.0)
        self.assertTrue(explicit_exact)
        self.assertEqual(omitted_score, 1.0)
        self.assertTrue(omitted_exact)


class EvaluateCaseTest(unittest.TestCase):
    """验证分层指标与错误分类。"""

    @staticmethod
    def sample() -> dict:
        """构造最小可执行测试样本。"""
        return {
            "id": "case_1",
            "track": "controlled_v1",
            "category": "unit",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "date": {"type": "string"},
                            },
                            "required": ["city", "date"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "expected_calls": [
                {"name": "weather", "arguments": {"city": ["北京"], "date": ["2026-10-20"]}}
            ],
            "executable": True,
        }

    def test_partial_argument_score(self) -> None:
        """工具正确但缺参时应保留部分参数得分。"""
        prediction = {
            "response": '<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call>'
        }
        result = evaluate_case(self.sample(), prediction)
        self.assertTrue(result["tool_correct"])
        self.assertEqual(result["argument_accuracy"], 0.5)
        self.assertFalse(result["execution_success"])
        self.assertEqual(result["error_type"], "argument_error")

    def test_wrong_tool_zeroes_argument_score(self) -> None:
        """工具选择错误时参数得分必须为零。"""
        prediction = {
            "response": '<tool_call>{"name":"search","arguments":{"city":"北京"}}</tool_call>'
        }
        result = evaluate_case(self.sample(), prediction)
        self.assertFalse(result["tool_correct"])
        self.assertEqual(result["argument_accuracy"], 0.0)
        self.assertEqual(result["error_type"], "tool_selection_error")


if __name__ == "__main__":
    unittest.main()
