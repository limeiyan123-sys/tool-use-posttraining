"""测试 Schema 鲁棒数据增强的关键不变量。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_robust_sft import function_calls, no_tool_difficulty, schema_alias_sample, validate_sample


def make_call_sample() -> dict:
    """构造一个含调用与干扰工具的最小样本。"""
    tools = [
        {
            "name": "weather_lookup",
            "description": "Get weather by city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "city name"},
                    "days": {"type": "integer", "description": "forecast days"},
                },
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        },
        {
            "name": "news_lookup",
            "description": "Get news",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    ]
    call = {"name": "weather_lookup", "arguments": {"city": "北京", "days": 2}}
    return {
        "conversations": [
            {"from": "human", "value": "查询北京两天天气"},
            {"from": "function_call", "value": json.dumps(call, ensure_ascii=False)},
        ],
        "tools": json.dumps(tools, ensure_ascii=False),
    }


class SchemaAliasTest(unittest.TestCase):
    """验证别名增强不会破坏调用与 Schema 的对应关系。"""

    def test_alias_is_consistent_and_removes_original_names(self) -> None:
        """工具名与参数名应同步变化，原名称不应残留在调用结构中。"""
        sample = make_call_sample()
        transformed = schema_alias_sample(sample, "unit-test", 7)
        validate_sample(transformed)
        call = function_calls(transformed)[0]
        self.assertTrue(call["name"].startswith("tool_"))
        self.assertTrue(all(name.startswith("arg_") for name in call["arguments"]))
        self.assertNotIn("weather_lookup", transformed["tools"])

    def test_alias_is_deterministic(self) -> None:
        """相同样本、盐值和种子必须得到逐字节一致的增强结果。"""
        sample = make_call_sample()
        self.assertEqual(
            schema_alias_sample(sample, "unit-test", 7),
            schema_alias_sample(sample, "unit-test", 7),
        )

    def test_parallel_call_list_is_supported(self) -> None:
        """同一消息内的并行调用列表应逐个一致重命名。"""
        sample = make_call_sample()
        call = json.loads(sample["conversations"][1]["value"])
        sample["conversations"][1]["value"] = json.dumps([call, call], ensure_ascii=False)
        transformed = schema_alias_sample(sample, "parallel-test", 7)
        validate_sample(transformed)
        calls = function_calls(transformed)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], calls[1]["name"])


class DifficultyTest(unittest.TestCase):
    """验证困难负例排序特征。"""

    def test_related_words_raise_negative_difficulty(self) -> None:
        """用户文本与候选工具描述重合时，难度分数应大于零。"""
        sample = make_call_sample()
        sample["conversations"] = [
            {"from": "human", "value": "Could you discuss weather patterns without looking them up?"},
            {"from": "gpt", "value": "I can discuss the general concept without a tool."},
        ]
        self.assertGreater(no_tool_difficulty(sample), 0.0)
        self.assertEqual(function_calls(sample), [])

    def test_empty_tool_negative_is_valid(self) -> None:
        """无候选工具的 no-tool 样本仍是合法训练数据，难度应为零。"""
        sample = {
            "conversations": [
                {"from": "human", "value": "查询一个当前不可用的服务"},
                {"from": "gpt", "value": "当前没有可用工具。"},
            ],
            "tools": "[]",
        }
        validate_sample(sample)
        self.assertEqual(no_tool_difficulty(sample), 0.0)


if __name__ == "__main__":
    unittest.main()
