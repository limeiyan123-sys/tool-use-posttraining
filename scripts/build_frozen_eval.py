#!/usr/bin/env python3
"""构建并冻结 Stage 1 评测集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


BFCL_CATEGORIES = {
    "simple_python": 20,
    "multiple": 20,
    "parallel": 10,
    "parallel_multiple": 10,
    "irrelevance": 10,
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建冻结评测集")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL 文件，遇到空行或非法 JSON 时立即失败。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 出现空行")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以稳定键顺序写入 JSONL 文件。"""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_select(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """按种子和样本 ID 的哈希稳定抽样，避免依赖输入文件顺序。"""
    if size > len(rows):
        raise ValueError(f"请求 {size} 条样本，但数据源只有 {len(rows)} 条")

    def rank(row: dict[str, Any]) -> str:
        sample_id = row.get("id")
        if not isinstance(sample_id, str):
            raise TypeError("BFCL 样本缺少字符串类型 id")
        return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()

    return sorted(rows, key=rank)[:size]


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """将 BFCL 的 dict/float 等类型映射为标准 JSON Schema 类型。"""
    result = json.loads(json.dumps(schema))
    type_mapping = {"dict": "object", "float": "number", "tuple": "array"}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if isinstance(node_type, str) and node_type in type_mapping:
                node["type"] = type_mapping[node_type]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(result)
    return result


def convert_ground_truth(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 BFCL ground truth 转成统一的候选值参数格式。"""
    calls: list[dict[str, Any]] = []
    for item in raw:
        if len(item) != 1:
            raise ValueError(f"ground truth 每项必须只有一个函数：{item}")
        name, arguments = next(iter(item.items()))
        if not isinstance(arguments, dict):
            raise TypeError(f"函数 {name} 的参数不是对象")
        allowed: dict[str, list[Any]] = {}
        for key, value in arguments.items():
            allowed[key] = value if isinstance(value, list) else [value]
        calls.append({"name": name, "arguments": allowed})
    return calls


def build_bfcl_lite(project_root: Path, seed: int) -> list[dict[str, Any]]:
    """从固定 BFCL 快照构建可复现的单轮子集。"""
    data_root = (
        project_root
        / "vendor"
        / "gorilla"
        / "berkeley-function-call-leaderboard"
        / "bfcl_eval"
        / "data"
    )
    answer_root = data_root / "possible_answer"
    output: list[dict[str, Any]] = []

    for category, size in BFCL_CATEGORIES.items():
        source_path = data_root / f"BFCL_v4_{category}.json"
        selected = stable_select(read_jsonl(source_path), size=size, seed=seed)
        answers: dict[str, list[dict[str, Any]]] = {}
        if category != "irrelevance":
            answer_path = answer_root / f"BFCL_v4_{category}.json"
            answers = {row["id"]: row["ground_truth"] for row in read_jsonl(answer_path)}

        for row in selected:
            sample_id = row["id"]
            turns = row["question"]
            if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], list):
                raise ValueError(f"{sample_id} 不是受支持的单轮 BFCL 格式")
            functions = [normalize_schema(function) for function in row["function"]]
            expected_calls = [] if category == "irrelevance" else convert_ground_truth(answers[sample_id])
            output.append(
                {
                    "id": f"bfcl_{sample_id}",
                    "source_id": sample_id,
                    "track": "bfcl_lite_v1",
                    "category": category,
                    "messages": turns[0],
                    "tools": [{"type": "function", "function": function} for function in functions],
                    "expected_calls": expected_calls,
                    "executable": False,
                }
            )
    return sorted(output, key=lambda row: row["id"])


def tool(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    """构造标准 OpenAI 风格函数工具定义。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def allowed(arguments: dict[str, Any]) -> dict[str, list[Any]]:
    """将唯一正确参数转换为统一候选值格式。"""
    return {key: [value] for key, value in arguments.items()}


def build_controlled() -> list[dict[str, Any]]:
    """构建可执行、无训练泄漏的受控评测集。"""
    string = lambda description: {"type": "string", "description": description}
    number = lambda description: {"type": "number", "description": description}
    integer = lambda description: {"type": "integer", "description": description}
    boolean = lambda description: {"type": "boolean", "description": description}

    catalog = {
        "weather_current": tool("weather_current", "查询城市当前天气。", {"city": string("城市")}, ["city"]),
        "weather_forecast": tool(
            "weather_forecast",
            "查询城市在指定日期的天气预报。",
            {"city": string("城市"), "date": string("ISO 日期")},
            ["city", "date"],
        ),
        "calculator": tool("calculator", "计算数学表达式。", {"expression": string("算术表达式")}, ["expression"]),
        "temperature_convert": tool(
            "temperature_convert",
            "在摄氏度与华氏度之间换算温度。",
            {
                "value": number("温度数值"),
                "from_unit": {"type": "string", "enum": ["C", "F"]},
                "to_unit": {"type": "string", "enum": ["C", "F"]},
            },
            ["value", "from_unit", "to_unit"],
        ),
        "currency_convert": tool(
            "currency_convert",
            "按固定测试汇率换算货币。",
            {
                "amount": number("金额"),
                "from_currency": {"type": "string", "enum": ["CNY", "USD", "EUR"]},
                "to_currency": {"type": "string", "enum": ["CNY", "USD", "EUR"]},
            },
            ["amount", "from_currency", "to_currency"],
        ),
        "parcel_status": tool("parcel_status", "查询包裹状态。", {"tracking_id": string("运单号")}, ["tracking_id"]),
        "inventory_lookup": tool(
            "inventory_lookup",
            "查询指定仓库的单个 SKU 库存。",
            {"sku": string("SKU"), "warehouse": string("仓库代码")},
            ["sku", "warehouse"],
        ),
        "batch_inventory_lookup": tool(
            "batch_inventory_lookup",
            "批量查询指定仓库的多个 SKU 库存。",
            {
                "skus": {"type": "array", "items": {"type": "string"}},
                "warehouse": string("仓库代码"),
            },
            ["skus", "warehouse"],
        ),
        "calendar_create": tool(
            "calendar_create",
            "创建日历事件。",
            {"title": string("标题"), "start_at": string("ISO 时间"), "duration_minutes": integer("持续分钟数")},
            ["title", "start_at", "duration_minutes"],
        ),
        "calendar_delete": tool("calendar_delete", "删除日历事件。", {"event_id": string("事件 ID")}, ["event_id"]),
        "calendar_list": tool(
            "calendar_list", "列出日期范围内的日历事件。", {"start_date": string("开始日期"), "end_date": string("结束日期")}, ["start_date", "end_date"]
        ),
        "flight_search": tool(
            "flight_search", "搜索航班。", {"origin": string("出发地"), "destination": string("目的地"), "date": string("日期")}, ["origin", "destination", "date"]
        ),
        "flight_status": tool("flight_status", "查询航班号的运行状态。", {"flight_number": string("航班号"), "date": string("日期")}, ["flight_number", "date"]),
        "route_drive": tool("route_drive", "规划驾车路线。", {"origin": string("起点"), "destination": string("终点")}, ["origin", "destination"]),
        "route_walk": tool("route_walk", "规划步行路线。", {"origin": string("起点"), "destination": string("终点")}, ["origin", "destination"]),
        "geo_distance": tool("geo_distance", "计算两地点直线距离。", {"origin": string("起点"), "destination": string("终点")}, ["origin", "destination"]),
        "product_search": tool("product_search", "按关键词搜索商品。", {"query": string("搜索词")}, ["query"]),
        "price_history": tool("price_history", "查询 SKU 的历史价格。", {"sku": string("SKU"), "days": integer("回溯天数")}, ["sku", "days"]),
        "translate_text": tool("translate_text", "翻译文本。", {"text": string("文本"), "target_language": string("目标语言")}, ["text", "target_language"]),
        "detect_language": tool("detect_language", "检测文本语言。", {"text": string("文本")}, ["text"]),
        "transliterate_text": tool("transliterate_text", "将文字转写为拉丁字母。", {"text": string("文本")}, ["text"]),
        "reserve_table": tool(
            "reserve_table",
            "预订餐桌。",
            {
                "restaurant": string("餐厅"),
                "party_size": integer("人数"),
                "time": string("ISO 时间"),
                "outdoor": boolean("是否室外座位"),
            },
            ["restaurant", "party_size", "time", "outdoor"],
        ),
        "create_ticket": tool(
            "create_ticket",
            "创建支持工单。",
            {
                "title": string("标题"),
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            ["title", "priority", "tags"],
        ),
        "hotel_search": tool(
            "hotel_search",
            "搜索酒店。",
            {
                "city": string("城市"),
                "check_in": string("入住日期"),
                "check_out": string("退房日期"),
                "guests": {
                    "type": "object",
                    "properties": {"adults": integer("成人数"), "children": integer("儿童数")},
                    "required": ["adults", "children"],
                    "additionalProperties": False,
                },
            },
            ["city", "check_in", "check_out", "guests"],
        ),
        "meteoroscope_v2": tool("meteoroscope_v2", "返回地点在目标日期的气象预测。", {"place": string("地点"), "target_day": string("日期")}, ["place", "target_day"]),
        "lexicon_bridge": tool("lexicon_bridge", "把输入内容转换为目标语言。", {"payload": string("输入内容"), "target_locale": string("目标语言代码")}, ["payload", "target_locale"]),
        "chrono_slotter": tool("chrono_slotter", "创建带时长的时间槽。", {"label": string("名称"), "start_at": string("开始时间"), "duration_minutes": integer("分钟")}, ["label", "start_at", "duration_minutes"]),
        "cargo_oracle": tool("cargo_oracle", "查询货运单状态，可选择返回历史轨迹。", {"parcel_code": string("货运单号"), "include_history": boolean("是否含历史")}, ["parcel_code", "include_history"]),
        "scalar_forge": tool("scalar_forge", "在两个温标之间换算数值。", {"value": number("数值"), "source_scale": {"type": "string", "enum": ["celsius", "fahrenheit"]}, "target_scale": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["value", "source_scale", "target_scale"]),
        "atlas_thread": tool("atlas_thread", "按指定交通方式规划起点到终点的路线。", {"origin": string("起点"), "destination": string("终点"), "mode": {"type": "string", "enum": ["walk", "drive"]}}, ["origin", "destination", "mode"]),
    }

    scenarios = [
        ("simple_001", "simple_selection", "查询北京在 2026-10-20 的天气预报。", ["weather_forecast"], "weather_forecast", {"city": "北京", "date": "2026-10-20"}),
        ("simple_002", "simple_selection", "计算 37 * 19。", ["calculator"], "calculator", {"expression": "37 * 19"}),
        ("simple_003", "simple_selection", "把 20 摄氏度换算为华氏度。", ["temperature_convert"], "temperature_convert", {"value": 20, "from_unit": "C", "to_unit": "F"}),
        ("simple_004", "simple_selection", "将 100 美元换算成人民币。", ["currency_convert"], "currency_convert", {"amount": 100, "from_currency": "USD", "to_currency": "CNY"}),
        ("simple_005", "simple_selection", "查询运单 SF1234567890 的状态。", ["parcel_status"], "parcel_status", {"tracking_id": "SF1234567890"}),
        ("simple_006", "simple_selection", "查询华北仓库 WH-N1 中 SKU A-104 的库存。", ["inventory_lookup"], "inventory_lookup", {"sku": "A-104", "warehouse": "WH-N1"}),
        ("competition_001", "tool_competition", "查上海 2026-10-22 的天气预报。", ["weather_current", "weather_forecast", "geo_distance"], "weather_forecast", {"city": "上海", "date": "2026-10-22"}),
        ("competition_002", "tool_competition", "在 2026-10-18 09:30 创建一个 45 分钟的项目复盘。", ["calendar_list", "calendar_delete", "calendar_create"], "calendar_create", {"title": "项目复盘", "start_at": "2026-10-18T09:30:00", "duration_minutes": 45}),
        ("competition_003", "tool_competition", "查询 CA1234 在 2026-10-21 的航班状态。", ["flight_search", "flight_status", "route_drive"], "flight_status", {"flight_number": "CA1234", "date": "2026-10-21"}),
        ("competition_004", "tool_competition", "规划从北京南站到故宫的步行路线。", ["route_drive", "route_walk", "geo_distance"], "route_walk", {"origin": "北京南站", "destination": "故宫"}),
        ("competition_005", "tool_competition", "查询 SKU B-778 过去 30 天的价格。", ["product_search", "inventory_lookup", "price_history"], "price_history", {"sku": "B-778", "days": 30}),
        ("competition_006", "tool_competition", "把“早上好”翻译成日语。", ["detect_language", "translate_text", "transliterate_text"], "translate_text", {"text": "早上好", "target_language": "日语"}),
        ("argument_001", "argument_constraints", "预订海棠餐厅 2026-10-19 18:30 的室外桌，共 4 人。", ["reserve_table"], "reserve_table", {"restaurant": "海棠餐厅", "party_size": 4, "time": "2026-10-19T18:30:00", "outdoor": True}),
        ("argument_002", "argument_constraints", "批量查询 WH-S2 仓库中 A-1、B-2 和 C-3 的库存。", ["batch_inventory_lookup"], "batch_inventory_lookup", {"skus": ["A-1", "B-2", "C-3"], "warehouse": "WH-S2"}),
        ("argument_003", "argument_constraints", "创建高优先级工单“登录超时”，标签为 auth 和 timeout。", ["create_ticket"], "create_ticket", {"title": "登录超时", "priority": "high", "tags": ["auth", "timeout"]}),
        ("argument_004", "argument_constraints", "把 98.6 华氏度转换为摄氏度。", ["temperature_convert"], "temperature_convert", {"value": 98.6, "from_unit": "F", "to_unit": "C"}),
        ("argument_005", "argument_constraints", "搜索杭州酒店：2026-11-01 入住，2026-11-03 退房，2 位成人和 1 名儿童。", ["hotel_search"], "hotel_search", {"city": "杭州", "check_in": "2026-11-01", "check_out": "2026-11-03", "guests": {"adults": 2, "children": 1}}),
        ("argument_006", "argument_constraints", "创建事件“模型评审”，开始时间 2026-10-25 14:00，持续 120 分钟。", ["calendar_create"], "calendar_create", {"title": "模型评审", "start_at": "2026-10-25T14:00:00", "duration_minutes": 120}),
        ("unseen_001", "unseen_schema", "用气象工具查询成都在 2026-10-30 的预测。", ["meteoroscope_v2", "cargo_oracle", "scalar_forge"], "meteoroscope_v2", {"place": "成都", "target_day": "2026-10-30"}),
        ("unseen_002", "unseen_schema", "将“谢谢”转换为法语，目标语言代码用 fr。", ["lexicon_bridge", "meteoroscope_v2", "atlas_thread"], "lexicon_bridge", {"payload": "谢谢", "target_locale": "fr"}),
        ("unseen_003", "unseen_schema", "创建名为“论文讨论”的时间槽：2026-10-26 10:00 开始，持续 60 分钟。", ["chrono_slotter", "cargo_oracle", "lexicon_bridge"], "chrono_slotter", {"label": "论文讨论", "start_at": "2026-10-26T10:00:00", "duration_minutes": 60}),
        ("unseen_004", "unseen_schema", "查询货运单 PKG-7788，并返回历史轨迹。", ["cargo_oracle", "chrono_slotter", "meteoroscope_v2"], "cargo_oracle", {"parcel_code": "PKG-7788", "include_history": True}),
        ("unseen_005", "unseen_schema", "使用标量换算工具将 32 华氏度转换为摄氏度。", ["scalar_forge", "atlas_thread", "cargo_oracle"], "scalar_forge", {"value": 32, "source_scale": "fahrenheit", "target_scale": "celsius"}),
        ("unseen_006", "unseen_schema", "规划从人民广场到外滩的驾车路线。", ["atlas_thread", "scalar_forge", "lexicon_bridge"], "atlas_thread", {"origin": "人民广场", "destination": "外滩", "mode": "drive"}),
        ("irrelevance_001", "irrelevance", "解释一下光合作用的基本过程。", ["weather_current", "calculator"], None, {}),
        ("irrelevance_002", "irrelevance", "写一首关于秋天的五言绝句。", ["calendar_create", "route_walk"], None, {}),
        ("irrelevance_003", "irrelevance", "你好，请简单介绍你自己。", ["parcel_status", "inventory_lookup"], None, {}),
        ("irrelevance_004", "irrelevance", "比较监督学习和无监督学习。", ["currency_convert", "flight_status"], None, {}),
    ]

    rows: list[dict[str, Any]] = []
    for sample_id, category, query, tool_names, expected_name, arguments in scenarios:
        expected_calls = [] if expected_name is None else [{"name": expected_name, "arguments": allowed(arguments)}]
        rows.append(
            {
                "id": f"controlled_{sample_id}",
                "track": "controlled_v1",
                "category": category,
                "messages": [{"role": "user", "content": query}],
                "tools": [catalog[name] for name in tool_names],
                "expected_calls": expected_calls,
                "executable": expected_name is not None,
            }
        )
    return rows


def git_commit(path: Path) -> str:
    """读取 Git 快照提交号，失败时抛出异常而不是猜测。"""
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    """构建数据集并写入不可覆盖的清单。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "data" / "evaluation" / "frozen_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    controlled_path = output_dir / "controlled_v1.jsonl"
    bfcl_path = output_dir / "bfcl_lite_v1.jsonl"
    manifest_path = output_dir / "manifest.json"

    existing = [path for path in (controlled_path, bfcl_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"冻结文件已存在，拒绝覆盖：{existing}")

    controlled = build_controlled()
    bfcl_lite = build_bfcl_lite(project_root, seed=args.seed)
    write_jsonl(controlled_path, controlled)
    write_jsonl(bfcl_path, bfcl_lite)

    gorilla_root = project_root / "vendor" / "gorilla"
    manifest = {
        "version": "frozen_v1",
        "seed": args.seed,
        "policy": "文件创建后禁止覆盖；如需修订必须创建新版本目录。",
        "gorilla_commit": git_commit(gorilla_root),
        "datasets": {
            controlled_path.name: {"samples": len(controlled), "sha256": sha256_file(controlled_path)},
            bfcl_path.name: {"samples": len(bfcl_lite), "sha256": sha256_file(bfcl_path)},
        },
        "bfcl_category_sizes": BFCL_CATEGORIES,
        "limitations": [
            "bfcl_lite_v1 是固定子集，不是 BFCL 官方排行榜成绩。",
            "受控集由本项目构造，主要用于消融、执行验证和未见 Schema 测试。",
            "公开 BFCL 样本可能已出现在基础模型预训练数据中，不能据此证明严格未见。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
