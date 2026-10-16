"""Stage 9 参数语义绑定数据构造测试。"""

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "stage9_schema_binding.py"
SPEC = importlib.util.spec_from_file_location("stage9_schema_binding", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_factorial_changes_only_identifier_and_order() -> None:
    """验证同一基础任务的四条件保持语义和 Gold 一致。"""
    family = MODULE.FAMILIES[0]
    rows = [MODULE.make_sample(family, 1, condition) for condition in MODULE.CONDITIONS]
    assert {row["condition"] for row in rows} == set(MODULE.CONDITIONS)
    assert len({str(row["messages"]) for row in rows}) == 1
    assert len({str(row["expected_semantic_arguments"]) for row in rows}) == 1
    assert rows[0]["semantic_to_actual"] == rows[2]["semantic_to_actual"]
    assert rows[1]["semantic_to_actual"] == rows[3]["semantic_to_actual"]
    assert list(rows[0]["tools"][0]["function"]["parameters"]["properties"]) == list(
        reversed(list(rows[2]["tools"][0]["function"]["parameters"]["properties"]))
    )


def test_opaque_identifiers_are_domain_neutral() -> None:
    """验证 B/D 使用固定的不透明参数名。"""
    row = MODULE.make_sample(MODULE.FAMILIES[1], 0, "B")
    assert list(row["actual_to_semantic"]) == ["arg_a", "arg_b", "arg_c", "arg_d"]
    assert row["expected_call"]["arguments"] == {
        "arg_a": "310000",
        "arg_b": "100000",
        "arg_c": 2.5,
    }


def test_description_alignment_is_deterministic() -> None:
    """验证描述驱动别名不依赖 Gold 值且结果稳定。"""
    row = MODULE.make_sample(MODULE.FAMILIES[2], 2, "B")
    first = MODULE.description_aliases(row)
    second = MODULE.description_aliases(row)
    assert first == second
    assert set(first) == {"arg_a", "arg_b", "arg_c", "arg_d"}
    assert len(set(first.values())) == 4


def test_roundtrip_preserves_order_and_prompt_semantics(tmp_path) -> None:
    """验证实际写入后读取的文件仍保留四条件字段顺序，防止排序抹除自变量。"""
    rows = [MODULE.make_sample(family, index, condition, formal=True) for family in MODULE.FAMILIES for index in range(4) for condition in MODULE.CONDITIONS]
    path = tmp_path / "data.jsonl"
    MODULE.write_jsonl(path, rows)
    loaded = MODULE.read_jsonl(path)
    assert MODULE.validate_factorial(loaded)["samples"] == 96
    corrupt = [json.loads(json.dumps(row, sort_keys=True)) for row in rows]
    import pytest
    with pytest.raises(RuntimeError, match="字段序列"):
        MODULE.validate_factorial(corrupt)


def test_formal_requests_are_disjoint_from_pilot() -> None:
    """验证正式请求与所有历史试运行请求无精确重复。"""
    pilot = {request for family in MODULE.FAMILIES for request, _ in family["requests"]}
    formal = {request for requests in MODULE.FORMAL_REQUESTS.values() for request, _ in requests}
    assert not pilot & formal
    assert len(formal) == 24


def test_boolean_numeric_confusion_is_not_correct() -> None:
    """验证布尔值 false 不会被数值 0 冒充正确，同时保护遗漏分类。"""
    import sys
    sys.path.insert(0, str(SCRIPT.parents[1]))
    from evaluation.metrics import parse_tool_calls
    row = MODULE.make_sample(MODULE.FAMILIES[1], 3, "B", formal=True)
    call = json.loads(json.dumps(row["expected_call"]))
    call["arguments"]["arg_d"] = 0
    case = MODULE.classify_case(row, {"response": "<tool_call>" + json.dumps(call) + "</tool_call>"}, parse_tool_calls)
    assert not case["decision_exact"]
    assert case["correct_fields"] == 3
    assert case["schema_errors"]


def test_refusal_is_distinct_from_broken_json() -> None:
    """复用既有无调用语义，拒绝回答不能误记为 JSON 解析故障。"""
    import sys
    sys.path.insert(0, str(SCRIPT.parents[1]))
    from evaluation.metrics import parse_tool_calls
    row = MODULE.make_sample(MODULE.FAMILIES[0], 0, "A", formal=True)
    refusal = MODULE.classify_case(row, {"response": "The current tool environment cannot complete this request.<|im_end|>"}, parse_tool_calls)
    broken = MODULE.classify_case(row, {"response": "<tool_call>{invalid}</tool_call>"}, parse_tool_calls)
    assert refusal["error_type"] == "false_rejection"
    assert refusal["parse_valid"]
    assert broken["error_type"] == "parsing_failure"
