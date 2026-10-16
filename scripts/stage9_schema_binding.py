#!/usr/bin/env python3
"""Stage 9：参数语义绑定的 2x2 因子诊断、推理和严格评测。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 20261016
CONDITIONS = ("A", "B", "C", "D")
FORMAL_REQUESTS = {
    "hotel_booking": (
        ("预订南京酒店，2027-06-11 入住 4 晚。", ("南京", "2027-06-11", 4, None)),
        ("预订苏州酒店，2027-07-09 入住 6 晚，需要 2 间房。", ("苏州", "2027-07-09", 6, 2)),
        ("预订昆明酒店，2027-08-18 入住 5 晚。", ("昆明", "2027-08-18", 5, None)),
        ("预订宁波酒店，2027-09-23 入住 7 晚，需要 3 间房。", ("宁波", "2027-09-23", 7, 3)),
    ),
    "shipment_quote": (
        ("查询从邮编 210000 寄到 650000 的 4.3 公斤包裹报价。", ("210000", "650000", 4.3, None)),
        ("算一下从 215000 到 330000 的 1.7 公斤快递，使用加急服务。", ("215000", "330000", 1.7, True)),
        ("从 400000 向 350000 寄 6.2 公斤包裹要多少钱？", ("400000", "350000", 6.2, None)),
        ("查询 450000 发往 315000 的 0.9 公斤包裹，明确不要加急。", ("450000", "315000", 0.9, False)),
    ),
    "sensor_report": (
        ("生成传感器 S-531 从 2027-06-10T07:00 到 2027-06-10T13:00 的报告。", ("S-531", "2027-06-10T07:00", "2027-06-10T13:00", None)),
        ("生成 S-642 在 2027-06-11T10:00 到 2027-06-11T15:30 的报告，按平均值聚合。", ("S-642", "2027-06-11T10:00", "2027-06-11T15:30", "mean")),
        ("查看传感器 S-753 从 2027-06-12T03:00 到 2027-06-12T09:00 的报告。", ("S-753", "2027-06-12T03:00", "2027-06-12T09:00", None)),
        ("生成 S-864 在 2027-06-13T16:00 到 2027-06-13T21:00 的报告，采用最大值聚合。", ("S-864", "2027-06-13T16:00", "2027-06-13T21:00", "max")),
    ),
    "catalog_filter": (
        ("筛选价格在 180 到 750 元之间的耳机。", ("耳机", 180, 750, None)),
        ("找 900 到 2800 元的投影仪，也显示缺货商品。", ("投影仪", 900, 2800, True)),
        ("筛选 80 到 460 元的鼠标。", ("鼠标", 80, 460, None)),
        ("找 600 到 1600 元的书桌，不要包含缺货商品。", ("书桌", 600, 1600, False)),
    ),
    "currency_alert": (
        ("当美元兑欧元汇率达到 0.93 时提醒我。", ("USD", "EUR", 0.93, None)),
        ("设置欧元兑人民币跌破 7.62 的提醒。", ("EUR", "CNY", 7.62, "below")),
        ("英镑兑美元达到 1.28 时通知我。", ("GBP", "USD", 1.28, None)),
        ("人民币兑日元升破 20.5 时提醒。", ("CNY", "JPY", 20.5, "above")),
    ),
    "meeting_room": (
        ("在智创楼 7 层找一间能坐 14 人的会议室。", ("智创楼", 7, 14, None)),
        ("在信息楼 4 层找容纳 9 人并支持视频会议的会议室。", ("信息楼", 4, 9, True)),
        ("在工程楼 10 层找一间 18 人会议室。", ("工程楼", 10, 18, None)),
        ("在材料楼 5 层找 6 人会议室，明确不需要视频设备。", ("材料楼", 5, 6, False)),
    ),
}
SYSTEM_PROMPT = (
    "你是严谨的工具调用助手。必须依据当前提供的工具 Schema 生成调用；"
    "不得猜测不存在的参数。需要调用时只输出一个结构化工具调用。"
)
FIELD_SYSTEM_PROMPT = (
    "你正在检查函数参数 Schema。根据参数描述判断哪个参数承载指定信息，"
    "只输出一个参数标识符，不要输出解释、引号或工具调用。"
)


FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "hotel_booking",
        "tool_name": "reserve_hotel_room",
        "tool_description": "按照住客要求预订酒店房间。",
        "fields": (
            ("destination_city", "string", "Destination city for the hotel stay."),
            ("check_in_date", "string", "Check-in date in YYYY-MM-DD format."),
            ("night_count", "integer", "Number of nights to reserve.") ,
            ("room_count", "integer", "Optional number of rooms when the user specifies it."),
        ),
        "required": ("destination_city", "check_in_date", "night_count"),
        "requests": (
            ("请预订杭州的酒店，2027-01-12 入住 3 晚。", ("杭州", "2027-01-12", 3, None)),
            ("帮我订成都酒店，2027-02-08 入住 2 晚，需要 2 间房。", ("成都", "2027-02-08", 2, 2)),
            ("预订青岛酒店，2027-03-17 入住 4 晚。", ("青岛", "2027-03-17", 4, None)),
            ("订一间厦门酒店，2027-04-21 入住 5 晚，需要 1 间房。", ("厦门", "2027-04-21", 5, 1)),
        ),
    },
    {
        "family": "shipment_quote",
        "tool_name": "quote_parcel_shipping",
        "tool_description": "计算两个邮编之间的包裹运输报价。",
        "fields": (
            ("origin_postcode", "string", "Origin postcode where the parcel is collected."),
            ("destination_postcode", "string", "Destination postcode where the parcel is delivered."),
            ("weight_kg", "number", "Parcel weight measured in kilograms."),
            ("priority_service", "boolean", "Optional priority-service flag when explicitly requested."),
        ),
        "required": ("origin_postcode", "destination_postcode", "weight_kg"),
        "requests": (
            ("查询从邮编 310000 寄到 100000 的 2.5 公斤包裹报价。", ("310000", "100000", 2.5, None)),
            ("算一下从 510000 到 200000 的 1.2 公斤快递，使用加急服务。", ("510000", "200000", 1.2, True)),
            ("从 430000 向 610000 寄 3.8 公斤包裹要多少钱？", ("430000", "610000", 3.8, None)),
            ("查询 250000 发往 710000 的 0.8 公斤包裹，明确不要加急。", ("250000", "710000", 0.8, False)),
        ),
    },
    {
        "family": "sensor_report",
        "tool_name": "generate_sensor_report",
        "tool_description": "为指定传感器和时间窗口生成聚合报告。",
        "fields": (
            ("sensor_id", "string", "Sensor identifier whose readings are requested."),
            ("window_start", "string", "Start timestamp of the reporting window."),
            ("window_end", "string", "End timestamp of the reporting window."),
            ("aggregation_method", "string", "Optional aggregation method explicitly requested by the user."),
        ),
        "required": ("sensor_id", "window_start", "window_end"),
        "requests": (
            ("生成传感器 S-104 从 2027-05-01T08:00 到 2027-05-01T12:00 的报告。", ("S-104", "2027-05-01T08:00", "2027-05-01T12:00", None)),
            ("生成 S-205 在 2027-05-02T09:30 到 2027-05-02T11:30 的报告，按平均值聚合。", ("S-205", "2027-05-02T09:30", "2027-05-02T11:30", "mean")),
            ("查看传感器 S-318 从 2027-05-03T00:00 到 2027-05-03T06:00 的报告。", ("S-318", "2027-05-03T00:00", "2027-05-03T06:00", None)),
            ("生成 S-427 在 2027-05-04T14:00 到 2027-05-04T18:00 的报告，采用最大值聚合。", ("S-427", "2027-05-04T14:00", "2027-05-04T18:00", "max")),
        ),
    },
    {
        "family": "catalog_filter",
        "tool_name": "filter_product_catalog",
        "tool_description": "按照品类和价格区间筛选商品目录。",
        "fields": (
            ("product_category", "string", "Product category to include in the catalog search."),
            ("minimum_price", "number", "Minimum allowed product price, inclusive."),
            ("maximum_price", "number", "Maximum allowed product price, inclusive."),
            ("include_out_of_stock", "boolean", "Optional flag controlling whether unavailable products are included."),
        ),
        "required": ("product_category", "minimum_price", "maximum_price"),
        "requests": (
            ("筛选价格在 200 到 800 元之间的机械键盘。", ("机械键盘", 200, 800, None)),
            ("找 1000 到 3500 元的显示器，也显示缺货商品。", ("显示器", 1000, 3500, True)),
            ("筛选 50 到 300 元的移动电源。", ("移动电源", 50, 300, None)),
            ("找 300 到 1200 元的办公椅，不要包含缺货商品。", ("办公椅", 300, 1200, False)),
        ),
    },
    {
        "family": "currency_alert",
        "tool_name": "create_exchange_rate_alert",
        "tool_description": "创建两个币种之间的汇率阈值提醒。",
        "fields": (
            ("base_currency", "string", "Base currency code used as the quoted unit."),
            ("quote_currency", "string", "Quote currency code used as the target unit."),
            ("rate_threshold", "number", "Exchange-rate threshold that triggers the alert."),
            ("crossing_direction", "string", "Optional threshold crossing direction explicitly requested."),
        ),
        "required": ("base_currency", "quote_currency", "rate_threshold"),
        "requests": (
            ("当美元兑人民币汇率达到 7.35 时提醒我。", ("USD", "CNY", 7.35, None)),
            ("设置欧元兑美元跌破 1.04 的提醒。", ("EUR", "USD", 1.04, "below")),
            ("日元兑人民币达到 0.052 时通知我。", ("JPY", "CNY", 0.052, None)),
            ("英镑兑欧元升破 1.22 时提醒。", ("GBP", "EUR", 1.22, "above")),
        ),
    },
    {
        "family": "meeting_room",
        "tool_name": "find_meeting_room",
        "tool_description": "按照位置和容量查找会议室。",
        "fields": (
            ("building_name", "string", "Building name where the meeting room must be located."),
            ("floor_number", "integer", "Floor number within the requested building."),
            ("attendee_count", "integer", "Number of attendees the room must accommodate."),
            ("video_required", "boolean", "Optional flag indicating whether video conferencing is required."),
        ),
        "required": ("building_name", "floor_number", "attendee_count"),
        "requests": (
            ("在创新楼 6 层找一间能坐 12 人的会议室。", ("创新楼", 6, 12, None)),
            ("在科研楼 3 层找容纳 8 人并支持视频会议的会议室。", ("科研楼", 3, 8, True)),
            ("在综合楼 11 层找一间 20 人会议室。", ("综合楼", 11, 20, None)),
            ("在实验楼 2 层找 5 人会议室，明确不需要视频设备。", ("实验楼", 2, 5, False)),
        ),
    },
)


def parse_args() -> argparse.Namespace:
    """解析 Stage 9 子命令参数。"""
    parser = argparse.ArgumentParser(description="Stage 9 参数语义绑定实验")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="构造并冻结 2x2 因子数据")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--tokenizer-path", type=Path, required=True)

    run = subparsers.add_parser("run", help="执行确定性推理")
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--dataset-name", default="factorial_v1.jsonl")
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--adapter-path", type=Path)
    run.add_argument("--output-path", type=Path, required=True)
    run.add_argument(
        "--strategy",
        choices=("direct", "explicit_mapping", "description_alignment"),
        default="direct",
    )
    run.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    run.add_argument("--device", choices=("auto", "cpu"), default="auto")
    run.add_argument("--cuda-memory-fraction", type=float, default=0.16)
    run.add_argument("--max-new-tokens", type=int, default=160)
    run.add_argument("--resume-partial", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="评测一个或多个模型预测")
    evaluate.add_argument("--project-root", type=Path, required=True)
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--prediction", action="append", required=True, help="name=path")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=5000)

    probe = subparsers.add_parser("build-field-probe", help="根据直接推理错误构造字段识别探针")
    probe.add_argument("--dataset-dir", type=Path, required=True)
    probe.add_argument("--cases", action="append", required=True)
    probe.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """以 UTF-8 创建 JSON 文件，拒绝静默覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以 UTF-8 创建 JSONL 文件，拒绝静默覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL，空行和非对象立即报错。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 出现空行")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(row)
    return rows


def make_property(field_type: str, description: str) -> dict[str, Any]:
    """构造无隐含默认值的扁平参数定义。"""
    prop: dict[str, Any] = {"type": field_type, "description": description}
    if field_type == "integer":
        prop["minimum"] = 1
    elif field_type == "number":
        prop["minimum"] = 0
    return prop


def opaque_names(field_count: int) -> list[str]:
    """生成固定长度、无领域含义的参数标识符。"""
    return [f"arg_{chr(ord('a') + index)}" for index in range(field_count)]


def condition_flags(condition: str) -> tuple[bool, bool]:
    """返回条件是否使用不透明标识符及是否反转字段顺序。"""
    if condition not in CONDITIONS:
        raise ValueError(f"未知条件：{condition}")
    return condition in {"B", "D"}, condition in {"C", "D"}


def make_sample(family: dict[str, Any], request_index: int, condition: str, formal: bool = False) -> dict[str, Any]:
    """从同一基础任务构造一个严格受控的 Schema 条件。"""
    use_opaque, reverse_order = condition_flags(condition)
    fields = list(family["fields"])
    actual_names = opaque_names(len(fields)) if use_opaque else [field[0] for field in fields]
    semantic_to_actual = dict(zip((field[0] for field in fields), actual_names))
    ordered_fields = list(reversed(fields)) if reverse_order else fields
    properties: dict[str, Any] = {}
    for semantic_name, field_type, description in ordered_fields:
        properties[semantic_to_actual[semantic_name]] = make_property(field_type, description)
        prop = properties[semantic_to_actual[semantic_name]]
        if semantic_name == "aggregation_method":
            prop["enum"] = ["mean", "max"]
            prop["description"] += " Use mean for average and max for maximum."
        if semantic_name == "crossing_direction":
            prop["enum"] = ["below", "above"]
            prop["description"] += " Use below for falling below and above for rising above."
        if semantic_name in {"base_currency", "quote_currency"}:
            prop["enum"] = ["USD", "EUR", "GBP", "CNY", "JPY"]
    required_semantic = list(family["required"])
    if reverse_order:
        required_semantic.reverse()
    required_actual = [semantic_to_actual[name] for name in required_semantic]
    request, values = (FORMAL_REQUESTS[family["family"]] if formal else family["requests"])[request_index]
    semantic_arguments = {
        semantic_name: value
        for (semantic_name, _, _), value in zip(fields, values)
        if value is not None
    }
    actual_arguments = {
        semantic_to_actual[name]: value for name, value in semantic_arguments.items()
    }
    optional_names = [field[0] for field in fields if field[0] not in family["required"]]
    optional_mentioned = any(name in semantic_arguments for name in optional_names)
    base_id = f"s9_{'test' if formal else 'pilot'}_{family['family']}_{request_index + 1:02d}"
    return {
        "id": f"{base_id}_{condition}",
        "base_id": base_id,
        "family": family["family"],
        "condition": condition,
        "identifier_mode": "opaque" if use_opaque else "semantic",
        "order_mode": "reversed" if reverse_order else "original",
        "messages": [{"role": "user", "content": request}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": family["tool_name"],
                    "description": family["tool_description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_actual,
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "semantic_to_actual": semantic_to_actual,
        "actual_to_semantic": {value: key for key, value in semantic_to_actual.items()},
        "field_types": {field[0]: field[1] for field in fields},
        "field_descriptions": {field[0]: properties[semantic_to_actual[field[0]]]["description"] for field in fields},
        "required_semantic": list(family["required"]),
        "optional_mentioned": optional_mentioned,
        "expected_call": {"name": family["tool_name"], "arguments": actual_arguments},
        "expected_semantic_arguments": semantic_arguments,
    }


def validate_factorial(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """验证四条件只改变标识符和字段序列，不改变任务语义。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["base_id"]].append(row)
    if len(grouped) != 24:
        raise RuntimeError(f"应有 24 个基础任务，实际 {len(grouped)}")
    for base_id, samples in grouped.items():
        by_condition = {sample["condition"]: sample for sample in samples}
        if set(by_condition) != set(CONDITIONS):
            raise RuntimeError(f"{base_id} 四条件不完整")
        reference = by_condition["A"]
        for sample in samples:
            if sample["messages"] != reference["messages"]:
                raise RuntimeError(f"{base_id} 用户请求发生变化")
            if sample["expected_semantic_arguments"] != reference["expected_semantic_arguments"]:
                raise RuntimeError(f"{base_id} Gold 语义发生变化")
            function = sample["tools"][0]["function"]
            reference_function = reference["tools"][0]["function"]
            if (function["name"], function["description"]) != (
                reference_function["name"],
                reference_function["description"],
            ):
                raise RuntimeError(f"{base_id} 工具信息发生变化")
            restored = {sample["actual_to_semantic"][name]: prop for name, prop in function["parameters"]["properties"].items()}
            if restored != reference_function["parameters"]["properties"]:
                raise RuntimeError(f"{base_id} 参数契约发生变化")
            expected_order = list(reference["semantic_to_actual"])
            if sample["condition"] in {"C", "D"}:
                expected_order.reverse()
            if list(restored) != expected_order:
                raise RuntimeError(f"{base_id} {sample['condition']} 字段序列不符合实验条件")
            restored_required = {sample["actual_to_semantic"][key] for key in function["parameters"]["required"]}
            if restored_required != set(reference["required_semantic"]):
                raise RuntimeError(f"{base_id} 必填集合发生变化")
    return {
        "base_tasks": len(grouped),
        "samples": len(rows),
        "conditions": Counter(row["condition"] for row in rows),
        "families": Counter(row["family"] for row in rows),
        "optional_mentioned_base_tasks": sum(
            samples[0]["optional_mentioned"] for samples in grouped.values()
        ),
    }


def build_data(args: argparse.Namespace) -> None:
    """构建 Pilot、完整因子数据并记录参数名分词长度。"""
    from transformers import AutoTokenizer

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    rows = [
        make_sample(family, request_index, condition, formal=True)
        for family in FAMILIES
        for request_index in range(4)
        for condition in CONDITIONS
    ]
    validation = validate_factorial(rows)
    pilot = [make_sample(family, 0, condition) for family in FAMILIES for condition in CONDITIONS]
    pilot_base_ids = {row["base_id"] for row in pilot}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path.resolve(), local_files_only=True)
    token_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["condition"] not in {"A", "B"}:
            continue
        for semantic_name, actual_name in row["semantic_to_actual"].items():
            token_rows.append(
                {
                    "base_id": row["base_id"],
                    "condition": row["condition"],
                    "semantic_name": semantic_name,
                    "actual_name": actual_name,
                    "token_ids": tokenizer.encode(actual_name, add_special_tokens=False),
                    "token_count": len(tokenizer.encode(actual_name, add_special_tokens=False)),
                }
            )
    write_jsonl(output_dir / "pilot_v1.jsonl", pilot)
    write_jsonl(output_dir / "factorial_v1.jsonl", rows)
    validate_factorial(read_jsonl(output_dir / "factorial_v1.jsonl"))
    rendered = []
    for row in read_jsonl(output_dir / "factorial_v1.jsonl"):
        messages, tools, _ = prepare_prompt(row, "direct")
        prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        properties = tools[0]["function"]["parameters"]["properties"]
        positions = [prompt.index(json.dumps(key) + ": {") for key in properties]
        if positions != sorted(positions):
            raise RuntimeError(f"{row['id']} 模板改变了 Schema 顺序")
        rendered.append({"id": row["id"], "prompt": prompt, "field_order": list(properties), "field_positions": positions})
    write_jsonl(output_dir / "rendered_prompts.jsonl", rendered)
    write_jsonl(output_dir / "identifier_tokenization.jsonl", token_rows)
    manifest = {
        "version": "stage9_schema_binding_factorial_v2",
        "seed": SEED,
        "design": "2x2: identifier semantic/opaque x order original/reversed",
        "validation": validation,
        "pilot_base_tasks": len(pilot_base_ids),
        "pilot_samples": len(pilot),
        "stage8_overlap_status": "新人工构造任务；不宣称基础模型预训练去污染",
        "pilot_test_request_disjoint": not ({row["messages"][0]["content"] for row in rows} & {row["messages"][0]["content"] for row in pilot}),
        "source_script_sha256": sha256_file(Path(__file__)),
        "construction": "人工定义的新请求与新工具家族；确定性脚本派生四条件",
        "files": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            manifest["files"][path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_dataset(dataset_dir: Path, dataset_name: str) -> list[dict[str, Any]]:
    """按冻结清单校验并加载数据。"""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    path = dataset_dir / dataset_name
    metadata = manifest["files"].get(dataset_name)
    if metadata is None or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"冻结数据哈希不匹配：{path}")
    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("数据包含重复 ID")
    return rows


def description_aliases(sample: dict[str, Any]) -> dict[str, str]:
    """以冻结、无模型的描述首词规则生成语义化别名。"""
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for actual_name, prop in sample["tools"][0]["function"]["parameters"]["properties"].items():
        stopwords = {"optional", "the", "of", "a", "an", "in", "for", "flag", "number"}
        words = [word for word in re.findall(r"[A-Za-z]+", prop["description"].lower()) if word not in stopwords][:3]
        if not words:
            raise RuntimeError(f"{sample['id']} 描述无法生成别名")
        alias = "_".join(words)
        base_alias = alias
        suffix = 2
        while alias in used:
            alias = f"{base_alias}_{suffix}"
            suffix += 1
        aliases[actual_name] = alias
        used.add(alias)
    return aliases


def prepare_prompt(sample: dict[str, Any], strategy: str) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, str]]:
    """按免训练策略准备消息、工具及输出名到原始名的映射。"""
    tools = copy.deepcopy(sample["tools"])
    output_to_actual = {name: name for name in sample["actual_to_semantic"]}
    system_prompt = FIELD_SYSTEM_PROMPT if sample.get("protocol") == "field_probe" else SYSTEM_PROMPT
    messages = [*sample["messages"]]
    if strategy == "explicit_mapping":
        mapping_lines = [
            f"{actual} = {semantic}: {sample['field_descriptions'][semantic]}"
            for semantic, actual in sample["semantic_to_actual"].items()
        ]
        system_prompt += "\n参数标识符映射如下：\n" + "\n".join(mapping_lines)
    elif strategy == "description_alignment":
        aliases = description_aliases(sample)
        function = tools[0]["function"]
        parameters = function["parameters"]
        parameters["properties"] = {
            aliases[name]: value for name, value in parameters["properties"].items()
        }
        parameters["required"] = [aliases[name] for name in parameters["required"]]
        output_to_actual = {alias: actual for actual, alias in aliases.items()}
    elif strategy != "direct":
        raise ValueError(f"未知策略：{strategy}")
    return [{"role": "system", "content": system_prompt}, *messages], tools, output_to_actual


def run_inference(args: argparse.Namespace) -> None:
    """加载基础模型或 LoRA，按固定贪心解码执行推理。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_dir = args.dataset_dir.resolve()
    rows = load_dataset(dataset_dir, args.dataset_name)
    rows = [row for row in rows if row.get("condition", "B") in set(args.conditions)]
    model_path = args.model_path.resolve()
    adapter_path = args.adapter_path.resolve() if args.adapter_path else None
    output_path = args.output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"预测文件已存在，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    existing: dict[str, dict[str, Any]] = {}
    mode = "x"
    if partial_path.exists():
        if not args.resume_partial:
            raise FileExistsError(f"发现临时结果，需显式续跑：{partial_path}")
        partial_rows = read_jsonl(partial_path)
        existing = {row["id"]: row for row in partial_rows}
        mode = "a"
        print(f"[WARNING] 从 {len(existing)} 条已完成预测续跑")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(SEED)
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None if args.device == "cpu" else "auto",
        local_files_only=True,
    )
    if args.device == "cpu":
        base_model.to("cpu")
    model = (
        PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
        if adapter_path
        else base_model
    )
    model.eval()
    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with partial_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(rows, start=1):
            if sample["id"] in existing:
                continue
            messages, tools, output_to_actual = prepare_prompt(sample, args.strategy)
            if sample.get("protocol") == "field_probe":
                messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n当前工具 Schema：\n" + json.dumps(tools, ensure_ascii=False)}
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            sample_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            record = {
                "id": sample["id"],
                "strategy": args.strategy,
                "response": tokenizer.decode(new_tokens, skip_special_tokens=False).strip(),
                "output_to_actual": output_to_actual,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(new_tokens.shape[0]),
                "latency_seconds": round(time.perf_counter() - sample_started, 6),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index == 1 or index % 12 == 0 or index == len(rows):
                print(f"已完成 {index}/{len(rows)}")
    partial_path.replace(output_path)
    metadata = {
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "dataset": args.dataset_name,
        "dataset_sha256": sha256_file(dataset_dir / args.dataset_name),
        "strategy": args.strategy,
        "conditions": args.conditions,
        "seed": SEED,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "samples": len(rows),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "cuda_memory_fraction": args.cuda_memory_fraction if args.device != "cpu" else None,
        "peak_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            if torch.cuda.is_available() and args.device != "cpu"
            else 0.0
        ),
        "prediction_sha256": sha256_file(output_path),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def normalize_call_arguments(call: dict[str, Any], sample: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """将策略输出参数名映射回统一语义空间。"""
    output_to_actual = prediction.get("output_to_actual", {})
    actual_to_semantic = sample["actual_to_semantic"]
    normalized: dict[str, Any] = {}
    for output_name, value in call.get("arguments", {}).items():
        actual_name = output_to_actual.get(output_name, output_name)
        semantic_name = actual_to_semantic.get(actual_name, f"__unknown__:{output_name}")
        normalized[semantic_name] = value
    return normalized


def classify_case(sample: dict[str, Any], prediction: dict[str, Any], parse_tool_calls: Any) -> dict[str, Any]:
    """严格解析调用并区分遗漏、错绑、额外字段与结构错误。"""
    from evaluation.metrics import values_equal, validate_arguments

    response = prediction.get("response")
    if not isinstance(response, str):
        raise TypeError(f"{sample['id']} 缺少字符串响应")
    parsed = parse_tool_calls(response)
    expected = sample["expected_semantic_arguments"]
    selected_tool: str | None = None
    normalized: dict[str, Any] = {}
    clean_response = response.removesuffix("<|im_end|>").strip()
    protocol_valid = bool(re.fullmatch(r"<tool_call>\s*\{.*\}\s*</tool_call>", clean_response, re.DOTALL))
    parse_valid = parsed.valid and (not parsed.calls or protocol_valid)
    one_call = parse_valid and len(parsed.calls) == 1
    if one_call:
        call = parsed.calls[0]
        selected_tool = call["name"]
        normalized = normalize_call_arguments(call, sample, prediction)
    tool_correct = one_call and selected_tool == sample["expected_call"]["name"]
    correct_fields = sum(tool_correct and name in normalized and values_equal(normalized[name], value) for name, value in expected.items())
    omitted = [name for name in expected if name not in normalized]
    wrong_value = [
        name for name, value in expected.items()
        if name in normalized and not values_equal(normalized[name], value)
    ]
    misbound = [
        name for name, value in expected.items()
        if not values_equal(normalized.get(name), value)
        and any(other_name != name and values_equal(other_value, value) for other_name, other_value in normalized.items())
    ]
    extra = [name for name in normalized if name not in expected]
    argument_exact = set(normalized) == set(expected) and all(values_equal(normalized[name], value) for name, value in expected.items())
    semantic_schema = copy.deepcopy(sample["tools"][0]["function"]["parameters"])
    semantic_schema["properties"] = {sample["actual_to_semantic"][key]: value for key, value in semantic_schema["properties"].items()}
    semantic_schema["required"] = sample["required_semantic"]
    schema_errors = validate_arguments(normalized, semantic_schema)
    decision_exact = bool(tool_correct and argument_exact and not schema_errors)
    misbinding_pairs = [(name, other_name) for name in misbound for other_name, other_value in normalized.items() if other_name != name and values_equal(other_value, expected[name])]
    same_type = [(left, right) for left, right in misbinding_pairs if sample["field_types"].get(left) == sample["field_types"].get(right)]
    cross_type = [(left, right) for left, right in misbinding_pairs if right in sample["field_types"] and sample["field_types"][left] != sample["field_types"][right]]
    swapped_pairs = [(left, right) for left, right in same_type if left < right and left in normalized and right in expected and values_equal(normalized[left], expected[right])]
    if not parse_valid:
        error_type = "parsing_failure"
    elif len(parsed.calls) == 0:
        error_type = "false_rejection"
    elif len(parsed.calls) != 1:
        error_type = "call_count_error"
    elif not tool_correct:
        error_type = "wrong_tool"
    elif omitted:
        error_type = "omission"
    elif misbound:
        error_type = "value_misbinding"
    elif wrong_value:
        error_type = "wrong_value"
    elif extra:
        error_type = "extra_argument"
    elif schema_errors:
        error_type = "schema_type_error"
    else:
        error_type = "none"
    return {
        "id": sample["id"],
        "base_id": sample["base_id"],
        "family": sample["family"],
        "condition": sample["condition"],
        "identifier_mode": sample["identifier_mode"],
        "order_mode": sample["order_mode"],
        "optional_mentioned": sample["optional_mentioned"],
        "response": response,
        "parse_valid": parse_valid,
        "parse_error": parsed.error,
        "schema_errors": schema_errors,
        "call_count": len(parsed.calls) if parsed.valid else None,
        "same_type_misbinding_pairs": same_type,
        "cross_type_misbinding_pairs": cross_type,
        "same_type_swaps": swapped_pairs,
        "optional_omitted": [name for name in omitted if name not in sample["required_semantic"]],
        "field_types": sample["field_types"],
        "required_semantic": sample["required_semantic"],
        "selected_tool": selected_tool,
        "tool_correct": tool_correct,
        "normalized_arguments": normalized,
        "expected_arguments": expected,
        "correct_fields": correct_fields,
        "expected_field_count": len(expected),
        "parameter_binding_accuracy": correct_fields / len(expected),
        "omitted_fields": omitted,
        "wrong_value_fields": wrong_value,
        "misbound_fields": misbound,
        "extra_fields": extra,
        "argument_exact": argument_exact,
        "decision_exact": decision_exact,
        "error_type": error_type,
    }


def mean(values: list[float | bool]) -> float:
    """计算非空序列均值。"""
    if not values:
        raise ValueError("不能计算空序列均值")
    return sum(values) / len(values)


def percentile(values: list[float], probability: float) -> float:
    """以最近秩规则计算分位数。"""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def condition_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合一个条件的调用与参数指标。"""
    total_expected_fields = sum(case["expected_field_count"] for case in cases)
    called = [case for case in cases if case["tool_correct"]]
    called_fields = sum(case["expected_field_count"] for case in called)
    return {
        "samples": len(cases),
        "decision_exact": mean([case["decision_exact"] for case in cases]),
        "tool_accuracy": mean([case["tool_correct"] for case in cases]),
        "argument_exact": mean([case["argument_exact"] for case in cases]),
        "parameter_binding_accuracy": sum(case["correct_fields"] for case in cases) / total_expected_fields,
        "omission_rate": sum(len(case["omitted_fields"]) for case in cases) / total_expected_fields,
        "value_misbinding_rate": sum(len(case["misbound_fields"]) for case in cases) / total_expected_fields,
        "parsing_failure_rate": mean([not case["parse_valid"] for case in cases]),
        "false_rejection_rate": mean([case["error_type"] == "false_rejection" for case in cases]),
        "binding_accuracy_given_correct_call": sum(case["correct_fields"] for case in called) / called_fields if called_fields else None,
        "omission_rate_given_correct_call": sum(len(case["omitted_fields"]) for case in called) / called_fields if called_fields else None,
        "correct_call_field_denominator": called_fields,
        "error_counts": dict(sorted(Counter(case["error_type"] for case in cases).items())),
        "same_type_misbinding_samples": sum(bool(case["same_type_misbinding_pairs"]) for case in cases),
        "same_type_swap_samples": sum(bool(case["same_type_swaps"]) for case in cases),
        "cross_type_misbinding_samples": sum(bool(case["cross_type_misbinding_pairs"]) for case in cases),
        "optional_required_by_user_omitted": sum(len(case["optional_omitted"]) for case in cases),
        "optional_required_by_user_total": sum(sum(name not in case["required_semantic"] for name in case["expected_arguments"]) for case in cases),
    }


def paired_bootstrap(cases: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    """按基础任务聚类重采样，估计因子主效应与交互项区间。"""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["base_id"]][case["condition"]] = case
    if any(set(group) != set(CONDITIONS) for group in grouped.values()):
        raise RuntimeError("四条件配对不完整，不能进行配对 Bootstrap")
    tasks = list(grouped.values())
    rng = random.Random(SEED)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        selected = [rng.choice(tasks) for _ in tasks]
        accuracies = {
            condition: mean([task[condition]["decision_exact"] for task in selected])
            for condition in CONDITIONS
        }
        bindings = {
            condition: sum(task[condition]["correct_fields"] for task in selected) / sum(task[condition]["expected_field_count"] for task in selected)
            for condition in CONDITIONS
        }
        draws["identifier_effect_decision"].append(accuracies["B"] - accuracies["A"])
        draws["order_effect_decision"].append(accuracies["C"] - accuracies["A"])
        draws["interaction_decision"].append(accuracies["D"] - accuracies["C"] - accuracies["B"] + accuracies["A"])
        draws["identifier_effect_binding"].append(bindings["B"] - bindings["A"])
        draws["order_effect_binding"].append(bindings["C"] - bindings["A"])
        draws["interaction_binding"].append(bindings["D"] - bindings["C"] - bindings["B"] + bindings["A"])
    return {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in draws.items()
    }


def aggregate_model(cases: list[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    """聚合模型的四条件、配对退化和分组结果。"""
    present_conditions = sorted({case["condition"] for case in cases})
    if present_conditions != list(CONDITIONS):
        return {"by_condition": {condition: condition_metrics([case for case in cases if case["condition"] == condition]) for condition in present_conditions}, "subset": True}
    by_condition = {
        condition: condition_metrics([case for case in cases if case["condition"] == condition])
        for condition in CONDITIONS
    }
    paired: dict[str, dict[str, int]] = {}
    task_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        task_map[case["base_id"]][case["condition"]] = case
    for target in ("B", "C", "D"):
        paired[target] = {
            "regressions_from_A": sum(task["A"]["decision_exact"] and not task[target]["decision_exact"] for task in task_map.values()),
            "recoveries_from_A": sum(not task["A"]["decision_exact"] and task[target]["decision_exact"] for task in task_map.values()),
        }
        paired[target]["regression_rate_all_tasks"] = paired[target]["regressions_from_A"] / len(task_map)
        paired[target]["recovery_rate_all_tasks"] = paired[target]["recoveries_from_A"] / len(task_map)
    effects = {
        "identifier_effect_decision": by_condition["B"]["decision_exact"] - by_condition["A"]["decision_exact"],
        "order_effect_decision": by_condition["C"]["decision_exact"] - by_condition["A"]["decision_exact"],
        "interaction_decision": (
            by_condition["D"]["decision_exact"] - by_condition["C"]["decision_exact"]
            - by_condition["B"]["decision_exact"] + by_condition["A"]["decision_exact"]
        ),
        "identifier_effect_binding": by_condition["B"]["parameter_binding_accuracy"] - by_condition["A"]["parameter_binding_accuracy"],
        "order_effect_binding": by_condition["C"]["parameter_binding_accuracy"] - by_condition["A"]["parameter_binding_accuracy"],
        "interaction_binding": (
            by_condition["D"]["parameter_binding_accuracy"] - by_condition["C"]["parameter_binding_accuracy"]
            - by_condition["B"]["parameter_binding_accuracy"] + by_condition["A"]["parameter_binding_accuracy"]
        ),
    }
    by_family = {
        family: {
            condition: condition_metrics([
                case for case in cases if case["family"] == family and case["condition"] == condition
            ])
            for condition in CONDITIONS
        }
        for family in sorted({case["family"] for case in cases})
    }
    by_optional = {
        label: {
            condition: condition_metrics([
                case for case in cases
                if case["optional_mentioned"] == flag and case["condition"] == condition
            ])
            for condition in CONDITIONS
        }
        for label, flag in (("optional_not_mentioned", False), ("optional_mentioned", True))
    }
    return {
        "by_condition": by_condition,
        "effects": effects,
        "paired_changes": paired,
        "bootstrap_95_ci": paired_bootstrap(cases, bootstrap_samples),
        "by_family": by_family,
        "by_optional": by_optional,
        "field_count": 4,
        "by_field_type": {
            field_type: {
                condition: {
                    "expected": sum(sum(case["field_types"][name] == field_type for name in case["expected_arguments"]) for case in cases if case["condition"] == condition),
                    "omitted": sum(sum(case["field_types"][name] == field_type for name in case["omitted_fields"]) for case in cases if case["condition"] == condition),
                }
                for condition in CONDITIONS
            }
            for field_type in ("string", "integer", "number", "boolean")
        },
    }


def parse_prediction_specs(specs: list[str]) -> dict[str, Path]:
    """解析 name=path 形式的预测参数。"""
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"预测参数必须为 name=path：{spec}")
        name, raw_path = spec.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"预测名称为空或重复：{name}")
        parsed[name] = Path(raw_path).resolve()
    return parsed


def render_report(summaries: dict[str, Any]) -> str:
    """渲染紧凑、可核查的模型对照报告。"""
    lines = [
        "# Stage 9 参数语义绑定因子实验结果",
        "",
        "| 模型/策略 | A Decision | B Decision | C Decision | D Decision | A Binding | B Binding | C Binding | D Binding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        metrics = summary["by_condition"]
        values = [metrics.get(c, {}).get("decision_exact") for c in CONDITIONS] + [metrics.get(c, {}).get("parameter_binding_accuracy") for c in CONDITIONS]
        lines.append(f"| {name} | " + " | ".join(f"{value * 100:.2f}%" if value is not None else "未运行" for value in values) + " |")
    lines.extend([
        "",
        "## 因子效应",
        "",
        "负值表示相对 A 退化；交互项为 D - C - B + A。",
        "",
        "| 模型/策略 | 标识符效应 | 顺序效应 | Decision 交互 | Binding 交互 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, summary in summaries.items():
        if "effects" not in summary:
            continue
        effects = summary["effects"]
        lines.append(
            f"| {name} | {effects['identifier_effect_decision'] * 100:+.2f} pp | "
            f"{effects['order_effect_decision'] * 100:+.2f} pp | "
            f"{effects['interaction_decision'] * 100:+.2f} pp | "
            f"{effects['interaction_binding'] * 100:+.2f} pp |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 24 个基础任务用于机制诊断，区间按基础任务聚类重采样，不把四个派生条件视为独立样本。",
        "- description_alignment 是固定描述抽取规则的 PA-Tool-inspired 简化对照，不等同于 PA-Tool 原方法。",
        "- explicit_mapping 提供额外语义与原顺序映射，仅作诊断条件，不保证构成性能上界。",
        "",
    ])
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> None:
    """统一评测多个模型或免训练策略并保存逐样本证据。"""
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    from evaluation.metrics import parse_tool_calls

    dataset_dir = args.dataset_dir.resolve()
    samples = load_dataset(dataset_dir, "factorial_v1.jsonl")
    sample_map = {sample["id"]: sample for sample in samples}
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    summaries: dict[str, Any] = {}
    for name, path in parse_prediction_specs(args.prediction).items():
        predictions = read_jsonl(path)
        prediction_map = {row["id"]: row for row in predictions}
        if len(prediction_map) != len(predictions) or not set(prediction_map) <= set(sample_map):
            raise RuntimeError(f"{name} 存在重复或未知预测 ID")
        conditions = {sample_map[key]["condition"] for key in prediction_map}
        expected_ids = {key for key, value in sample_map.items() if value["condition"] in conditions}
        if set(prediction_map) != expected_ids:
            missing = sorted(expected_ids - set(prediction_map))
            extra = sorted(set(prediction_map) - set(sample_map))
            raise RuntimeError(f"{name} 预测 ID 不完整，missing={missing[:3]}, extra={extra[:3]}")
        cases = [classify_case(sample, prediction_map[sample["id"]], parse_tool_calls) for sample in samples if sample["id"] in prediction_map]
        summary = aggregate_model(cases, args.bootstrap_samples)
        summaries[name] = summary
        write_jsonl(output_dir / f"{name}_cases.jsonl", cases)
        write_json(output_dir / f"{name}_summary.json", summary)
    write_json(output_dir / "comparison.json", summaries)
    (output_dir / "comparison.md").write_text(render_report(summaries), encoding="utf-8")
    print(render_report(summaries))


def build_field_probe(args: argparse.Namespace) -> None:
    """从 A 正确但 B/D 退化的任务中构造字段识别对照。"""
    dataset_dir = args.dataset_dir.resolve()
    rows = load_dataset(dataset_dir, "factorial_v1.jsonl")
    sample_map = {row["id"]: row for row in rows}
    failing_ids: set[str] = set()
    target_fields: dict[str, set[str]] = defaultdict(set)
    control_ids: set[str] = set()
    for case_path in args.cases:
        cases = read_jsonl(Path(case_path).resolve())
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for case in cases:
            grouped[case["base_id"]][case["condition"]] = case
        for base_id, group in grouped.items():
            for condition in ("B", "C", "D"):
                for field in group[condition]["omitted_fields"] + group[condition]["wrong_value_fields"]:
                    if field not in group["A"]["omitted_fields"] + group["A"]["wrong_value_fields"]:
                        failing_ids.add(base_id)
                        target_fields[base_id].add(field)
            if all(group[condition]["parameter_binding_accuracy"] == 1 for condition in CONDITIONS):
                control_ids.add(base_id)
    selected_failures = sorted(failing_ids)[:6]
    selected_controls = sorted(control_ids - set(selected_failures))[:6]
    selected = [(base_id, "regression") for base_id in selected_failures] + [
        (base_id, "control") for base_id in selected_controls
    ]
    if not selected_failures:
        raise RuntimeError("未发现 A 正确而 B/D 退化的任务，不应运行字段识别探针")
    probe_rows: list[dict[str, Any]] = []
    for base_id, group_label in selected:
        for condition in CONDITIONS:
            source = sample_map[f"{base_id}_{condition}"]
            candidates = list(source["expected_semantic_arguments"])
            target_semantic = sorted(target_fields[base_id])[0] if group_label == "regression" else candidates[-1]
            target_actual = source["semantic_to_actual"][target_semantic]
            probe = copy.deepcopy(source)
            probe["id"] = f"{source['id']}_field_probe"
            probe["protocol"] = "field_probe"
            target_value = source["expected_semantic_arguments"][target_semantic]
            labels = {"destination_city": "入住城市", "check_in_date": "入住日期", "night_count": "入住晚数", "room_count": "房间数量", "origin_postcode": "寄出邮编", "destination_postcode": "收件邮编", "weight_kg": "包裹公斤重量", "priority_service": "是否加急", "sensor_id": "传感器编号", "window_start": "报告开始时间", "window_end": "报告结束时间", "aggregation_method": "聚合方法", "product_category": "商品类别", "minimum_price": "最低价格", "maximum_price": "最高价格", "include_out_of_stock": "是否包含缺货商品", "base_currency": "汇率的基准币种", "quote_currency": "汇率的目标币种", "rate_threshold": "触发提醒的汇率值", "crossing_direction": "穿越阈值的方向", "building_name": "会议室所在楼宇", "floor_number": "会议室所在楼层", "attendee_count": "会议参加人数", "video_required": "是否需要视频会议设备"}
            probe["messages"] = [{
                "role": "user",
                "content": (
                    f"原始用户请求：{source['messages'][0]['content']}\n"
                    f"明确的用户条件是：{labels[target_semantic]}为 {json.dumps(target_value, ensure_ascii=False)}。"
                    "请只回答承载这个条件的参数 ID。"
                ),
            }]
            probe["probe_target_semantic"] = target_semantic
            probe["probe_target_actual"] = target_actual
            probe["probe_group"] = group_label
            probe_rows.append(probe)
    output_path = args.output_path.resolve()
    write_jsonl(output_path, probe_rows)
    write_json(output_path.parent / "manifest.json", {"version": "stage9_field_probe_v1", "selection": "三个模型配对错误的并集，字典序选最多六个退化任务及六个无绑定错误控制；诊断性选择，不代表泛化评估", "failures": selected_failures, "controls": selected_controls, "files": {output_path.name: {"sha256": sha256_file(output_path)}}})
    print(json.dumps({"samples": len(probe_rows), "failures": selected_failures, "controls": selected_controls}, ensure_ascii=False, indent=2))


def main() -> None:
    """分派 Stage 9 子命令。"""
    args = parse_args()
    if args.command == "build":
        build_data(args)
    elif args.command == "run":
        run_inference(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "build-field-probe":
        build_field_probe(args)
    else:
        raise RuntimeError(f"未处理的子命令：{args.command}")


if __name__ == "__main__":
    main()
