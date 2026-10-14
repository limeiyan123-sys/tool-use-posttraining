"""反事实工具世界构造与指标单元测试。"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluation.tool_world_metrics import aggregate_tool_world, evaluate_tool_world_case
from scripts.build_counterfactual_sft import world_to_sharegpt
from scripts.build_failure_weighted_sft import build_weighted_worlds
from scripts.build_tool_world_eval import (
    build_capability_removal_world,
    build_distractor_world,
    build_ephemeral_world,
    build_original_world,
    build_substitution_world,
    expected_calls_fit_tools,
)


def base_task() -> dict:
    """构造包含两个独立调用的最小基础任务。"""
    return {
        "source_id": "unit_1",
        "category": "unit",
        "messages": [{"role": "user", "content": "查询北京天气并计算 1+2"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "查询指定城市天气。",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "城市"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "计算两个整数之和。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
        "expected_calls": [
            {"name": "weather", "arguments": {"city": ["北京"]}},
            {"name": "add", "arguments": {"a": [1], "b": [2]}},
        ],
    }


def prediction_for(world: dict) -> dict:
    """把候选值标准答案转换为一条严格工具调用预测。"""
    blocks = []
    for call in world["expected_calls"]:
        arguments = {key: values[0] for key, values in call["arguments"].items()}
        import json

        blocks.append(
            "<tool_call>"
            + json.dumps({"name": call["name"], "arguments": arguments}, ensure_ascii=False)
            + "</tool_call>"
        )
    return {"response": "\n".join(blocks) if blocks else "当前工具无法完成该请求。"}


class ToolWorldConstructionTest(unittest.TestCase):
    """验证五类工具世界的语义与 Gold 更新。"""

    def test_ephemeral_interface_is_semantically_equivalent(self) -> None:
        """匿名接口预测映射回规范接口后应精确正确。"""
        world = build_ephemeral_world(base_task(), seed=7)
        result = evaluate_tool_world_case(world, prediction_for(world))
        self.assertTrue(result["decision_exact"])
        self.assertTrue(result["semantic_exact"])

    def test_removal_keeps_independent_call(self) -> None:
        """移除天气工具后仍应保留独立的加法调用。"""
        world = build_capability_removal_world(base_task(), "weather", seed=7)
        self.assertEqual([call["name"] for call in world["expected_calls"]], ["add"])

    def test_substitution_recomputes_gold(self) -> None:
        """不兼容替换工具不应被机械写回标准答案。"""
        world = build_substitution_world(base_task(), "weather", "city", "上海", seed=7)
        replacement = world["transformation"]["replacement_tool"]
        self.assertNotIn(replacement, [call["name"] for call in world["expected_calls"]])
        replacement_tool = next(
            tool for tool in world["tools"] if tool["function"]["name"] == replacement
        )
        self.assertEqual(
            replacement_tool["function"]["parameters"]["properties"]["city"]["enum"],
            ["上海"],
        )

    def test_distractor_does_not_change_gold(self) -> None:
        """加入不适用干扰工具后标准答案应保持原调用。"""
        world = build_distractor_world(base_task(), "weather", "city", "上海", seed=7)
        self.assertEqual(world["expected_calls"], base_task()["expected_calls"])

    def test_undeclared_gold_argument_is_rejected(self) -> None:
        """标准答案含未声明参数时不得进入训练或评测数据。"""
        base = base_task()
        base["expected_calls"][0]["arguments"]["type"] = ["hourly"]
        self.assertFalse(expected_calls_fit_tools(base["expected_calls"], base["tools"]))


class ToolWorldMetricTest(unittest.TestCase):
    """验证成对与全世界组指标。"""

    def test_all_correct_group(self) -> None:
        """五个世界均正确时所有成对指标应为一。"""
        base = base_task()
        worlds = [
            build_original_world(base),
            build_ephemeral_world(base, seed=7),
            build_capability_removal_world(base, "weather", seed=7),
            build_substitution_world(base, "weather", "city", "上海", seed=7),
            build_distractor_world(base, "weather", "city", "上海", seed=7),
        ]
        results = [evaluate_tool_world_case(world, prediction_for(world)) for world in worlds]
        summary = aggregate_tool_world(results)
        self.assertEqual(summary["interface_equivariance_accuracy"], 1.0)
        self.assertEqual(summary["all_worlds_group_accuracy"]["accuracy"], 1.0)
        self.assertTrue(
            all(
                metrics["accuracy"] == 1.0
                for metrics in summary["counterfactual_pair_accuracy"].values()
            )
        )


class CounterfactualSftConversionTest(unittest.TestCase):
    """验证工具世界到训练样本的严格转换。"""

    def test_function_calls_choose_first_gold_candidate(self) -> None:
        """函数调用训练目标应确定性采用首个 BFCL 候选值。"""
        sample = world_to_sharegpt(build_original_world(base_task()))
        import json

        value = json.loads(sample["conversations"][1]["value"])
        self.assertEqual(value[0]["arguments"]["city"], "北京")
        self.assertEqual(value[1]["arguments"], {"a": 1, "b": 2})

    def test_no_call_world_uses_explicit_refusal(self) -> None:
        """能力完全移除后应生成无 function_call 的显式拒绝监督。"""
        single = base_task()
        single["expected_calls"] = [single["expected_calls"][0]]
        single["tools"] = [single["tools"][0]]
        world = build_capability_removal_world(single, "weather", seed=7)
        sample = world_to_sharegpt(world)
        self.assertEqual(sample["conversations"][1]["from"], "gpt")
        self.assertIn("cannot complete", sample["conversations"][1]["value"])

    def test_failure_weighted_budget(self) -> None:
        """失败加权方法必须保持 400 条总预算和预定世界分配。"""
        pool = []
        for index in range(250):
            sample = deepcopy(base_task())
            sample["source_id"] = f"unit_{index:03d}"
            sample["target"] = ("weather", "city", "上海")
            pool.append(sample)
        worlds = build_weighted_worlds(pool, seed=7)
        self.assertEqual(len(worlds), 400)
        self.assertEqual(
            Counter(world["world_type"] for world in worlds),
            {
                "original": 150,
                "capability_substitution": 150,
                "capability_removal": 50,
                "ephemeral_interface": 25,
                "distractor": 25,
            },
        )


if __name__ == "__main__":
    unittest.main()
