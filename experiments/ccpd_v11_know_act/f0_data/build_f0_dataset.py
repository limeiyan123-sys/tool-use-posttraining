#!/usr/bin/env python3
"""从固定 BFCL 快照构造 CCPD-v1.1 的严格合同配对数据。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple")
CONSTRAINT_KEYS = {"enum", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "const", "pattern"}


def parse_args() -> argparse.Namespace:
    """解析数据路径、随机种子与目标划分规模。"""
    parser = argparse.ArgumentParser(description="构造 CCPD-v1.1 F0 冻结合同配对")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--gorilla-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--dev", type=int, default=4)
    parser.add_argument("--test-id", type=int, default=10)
    parser.add_argument("--test-ood-family", type=int, default=25)
    parser.add_argument("--test-ood-constraint", type=int, default=4)
    return parser.parse_args()


def stable_digest(*parts: object) -> str:
    """生成与输入顺序无关的稳定十六进制摘要。"""
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL，并在空行或非对象记录处失败。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 出现空行")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """以稳定键顺序写出 JSONL，拒绝静默覆盖。"""
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已冻结文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """计算文件内容的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """把 BFCL 的扩展类型名映射为标准 JSON Schema 类型。"""
    result = copy.deepcopy(schema)
    mapping = {"dict": "object", "float": "number", "tuple": "array"}

    def walk(node: Any) -> None:
        """递归规范化嵌套 Schema。"""
        if isinstance(node, dict):
            node_type = node.get("type")
            if isinstance(node_type, str) and node_type in mapping:
                node["type"] = mapping[node_type]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(result)
    return result


def convert_ground_truth(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 BFCL Ground Truth 统一为候选值列表格式。"""
    calls: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"非法 Ground Truth 项：{item!r}")
        name, arguments = next(iter(item.items()))
        if not isinstance(arguments, dict):
            raise TypeError(f"函数 {name} 的参数不是对象")
        calls.append(
            {
                "name": name,
                "arguments": {
                    key: value if isinstance(value, list) else [value]
                    for key, value in arguments.items()
                },
            }
        )
    return calls


def collect_source_ids(value: Any, output: set[str]) -> None:
    """递归收集历史数据中的 source_id，用于暴露边界审计。"""
    if isinstance(value, dict):
        source_id = value.get("source_id")
        if isinstance(source_id, str):
            output.add(source_id)
        for child in value.values():
            collect_source_ids(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_source_ids(child, output)


def load_historical_source_ids(project_root: Path) -> set[str]:
    """从既有 data 目录中收集所有已使用 BFCL 来源 ID。"""
    result: set[str] = set()
    data_root = project_root / "data"
    for path in sorted(data_root.rglob("*.jsonl")):
        for row in read_jsonl(path):
            collect_source_ids(row, result)
    for path in sorted(data_root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        collect_source_ids(value, result)
    return result


def scalar_in_query(value: Any, query: str) -> bool:
    """保守检查标量 Gold 是否在请求中具有直接、可定位证据。"""
    if value is None or isinstance(value, bool):
        return False
    normalized = query.casefold()
    if isinstance(value, (int, float)):
        rendered = format(value, "g")
        return re.search(rf"(?<![0-9.]){re.escape(rendered)}(?![0-9.])", normalized) is not None
    if isinstance(value, str):
        stripped = value.strip().casefold()
        if len(stripped) < 2:
            return False
        return stripped in normalized
    return False


def query_template(query: str, target_value: Any) -> str:
    """生成用于跨划分去重的保守请求模板。"""
    normalized = query.casefold()
    rendered = str(target_value).casefold()
    if rendered:
        normalized = normalized.replace(rendered, " <target> ")
    normalized = re.sub(r"['\"][^'\"]+['\"]", " <quoted> ", normalized)
    normalized = re.sub(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?![a-z])", " <num> ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def materialize_arguments(arguments: dict[str, Any]) -> dict[str, Any] | None:
    """从候选值列表生成确定参数；复杂或空候选直接舍弃。"""
    output: dict[str, Any] = {}
    for name, candidates in arguments.items():
        if not isinstance(candidates, list) or not candidates:
            return None
        value = candidates[0]
        if isinstance(value, (dict, list)) or value is None:
            return None
        output[name] = value
    return output


def alternative_string(value: str, salt: str) -> str:
    """构造与 Gold 等长且形态相似的稳定伪值，避免显眼前缀捷径。"""
    digest = stable_digest(salt, value)
    letters = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    output: list[str] = []
    cursor = 0
    for character in value:
        code = int(digest[cursor % len(digest)], 16)
        cursor += 1
        if character.isalpha():
            replacement = letters[code % len(letters)]
            output.append(replacement.upper() if character.isupper() else replacement)
        elif character.isdigit():
            output.append(digits[code % len(digits)])
        else:
            output.append(character)
    candidate = "".join(output)
    if candidate.casefold() == value.casefold():
        for index, character in enumerate(candidate):
            if character.isalpha():
                replacement = "z" if character.casefold() != "z" else "y"
                candidate = candidate[:index] + (replacement.upper() if character.isupper() else replacement) + candidate[index + 1 :]
                break
        else:
            raise ValueError("无法为不含字母的字符串生成可靠替代值")
    return candidate


def build_constraints(value: Any, salt: str, family: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """构造同类别 E+/E- 约束，并在数值任务中平衡上下界方向。"""
    if family == "unseen_const":
        if isinstance(value, str):
            negative = alternative_string(value, f"{salt}:const")
        elif isinstance(value, int) and not isinstance(value, bool):
            negative = value + 1
        elif isinstance(value, float):
            negative = value + max(1.0, abs(value) * 0.1)
        else:
            raise ValueError("const 仅支持字符串和数值标量")
        return {"const": value}, {"const": negative}, "const"
    if isinstance(value, str):
        negative = alternative_string(value, f"{salt}:minus")
        return {"enum": [value]}, {"enum": [negative]}, "enum"
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("数值范围仅支持有限整数或浮点数")
    if isinstance(value, int):
        step: int | float = max(1, abs(value) // 10 or 1)
    else:
        step = max(1.0, abs(value) * 0.1)
    if int(stable_digest(salt, "range_direction")[:2], 16) % 2:
        return {"minimum": value}, {"minimum": value + step}, "numeric_minimum"
    return {"maximum": value}, {"maximum": value - step}, "numeric_maximum"


def render_constraint(constraint: dict[str, Any]) -> str:
    """以统一模板将机器可执行约束同步写入自然语言描述。"""
    key, value = next(iter(constraint.items()))
    if key == "enum":
        rendered = ", ".join(json.dumps(item, ensure_ascii=False) for item in value)
        return f"The contract permits exactly these values for this field: {rendered}."
    if key == "maximum":
        return f"The contract permits values no greater than {value}."
    if key == "minimum":
        return f"The contract permits values no smaller than {value}."
    if key == "const":
        return f"The contract permits exactly this value: {json.dumps(value, ensure_ascii=False)}."
    raise ValueError(f"不支持的约束：{constraint}")


def apply_constraint(
    source_tool: dict[str, Any], opaque_name: str, target: str, constraint: dict[str, Any]
) -> dict[str, Any]:
    """复制原工具并仅替换目标字段的合同约束。"""
    tool = copy.deepcopy(source_tool)
    function = tool["function"]
    function["name"] = opaque_name
    field = function["parameters"]["properties"][target]
    for key in CONSTRAINT_KEYS:
        field.pop(key, None)
    clause = render_constraint(constraint)
    function["description"] = f"{function.get('description', '').rstrip()} {clause}".strip()
    field["description"] = f"{field.get('description', '').rstrip()} {clause}".strip()
    field.update(copy.deepcopy(constraint))
    return tool


def remove_constraint(tool: dict[str, Any], target: str, constraint: dict[str, Any]) -> dict[str, Any]:
    """移除已知约束键及其追加描述，恢复同一中性工具副本。"""
    result = copy.deepcopy(tool)
    function = result["function"]
    field = function["parameters"]["properties"][target]
    clause = render_constraint(constraint)
    for container in (function, field):
        description = container.get("description", "").rstrip()
        suffix = f" {clause}"
        if not description.endswith(suffix):
            raise ValueError("无法从工具描述中可靠移除旧合同约束")
        container["description"] = description[: -len(suffix)].rstrip()
    for key in CONSTRAINT_KEYS:
        field.pop(key, None)
    return result


def schema_accepts(arguments: dict[str, Any], tool: dict[str, Any]) -> bool:
    """执行本实验覆盖的 required、enum、范围和 const 合同。"""
    parameters = tool["function"]["parameters"]
    properties = parameters.get("properties", {})
    if any(name not in arguments for name in parameters.get("required", [])):
        return False
    if set(arguments) - set(properties):
        return False
    for name, value in arguments.items():
        schema = properties[name]
        if "enum" in schema and value not in schema["enum"]:
            return False
        if "const" in schema and value != schema["const"]:
            return False
        if "maximum" in schema and (isinstance(value, bool) or value > schema["maximum"]):
            return False
        if "minimum" in schema and (isinstance(value, bool) or value < schema["minimum"]):
            return False
    return True


def select_target(
    query: str, expected_call: dict[str, Any], source_tool: dict[str, Any]
) -> tuple[str, Any, dict[str, Any]] | None:
    """选择具有请求证据、必填且无默认值的标量目标字段。"""
    parameters = source_tool["function"]["parameters"]
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    arguments = materialize_arguments(expected_call["arguments"])
    if arguments is None:
        return None
    candidates: list[tuple[str, Any]] = []
    for name, value in arguments.items():
        schema = properties.get(name)
        if not isinstance(schema, dict) or name not in required or "default" in schema:
            continue
        if schema.get("type") not in {"string", "integer", "number"}:
            continue
        if scalar_in_query(value, query):
            candidates.append((name, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: stable_digest(expected_call["name"], item[0]))
    target, value = candidates[0]
    return target, value, arguments


def build_candidate(
    row: dict[str, Any], category: str, ground_truth: list[dict[str, Any]], historical_ids: set[str]
) -> tuple[dict[str, Any] | None, str]:
    """从一个 BFCL 单轮任务构造严格闭世界合同对。"""
    turns = row.get("question")
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], list):
        return None, "not_single_turn"
    messages = turns[0]
    if len(messages) != 1 or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
        return None, "not_single_user_message"
    try:
        expected_calls = convert_ground_truth(ground_truth)
    except (TypeError, ValueError):
        return None, "invalid_ground_truth"
    if len(expected_calls) != 1:
        return None, "not_single_call"
    call = expected_calls[0]
    tools = [
        {"type": "function", "function": normalize_schema(function)}
        for function in row.get("function", [])
    ]
    matches = [tool for tool in tools if tool["function"].get("name") == call["name"]]
    if len(matches) != 1:
        return None, "target_tool_not_unique"
    query = messages[0]["content"]
    selected = select_target(query, call, matches[0])
    if selected is None:
        return None, "no_strict_target"
    target, value, arguments = selected
    if re.search(rf"(?:under|less\s+than|below)\s+{re.escape(format(value, 'g'))}(?![0-9.])", query.casefold()) if isinstance(value, (int, float)) else False:
        return None, "strict_boundary_ambiguity"
    source_id = row.get("id")
    if not isinstance(source_id, str):
        return None, "missing_source_id"
    try:
        positive, negative, constraint_family = build_constraints(value, source_id, "seen")
    except ValueError:
        return None, "unsupported_value"
    opaque_name = f"tool_{stable_digest('f0', source_id)[:12]}"
    plus_tool = apply_constraint(matches[0], opaque_name, target, positive)
    minus_tool = apply_constraint(matches[0], opaque_name, target, negative)
    if not schema_accepts(arguments, plus_tool) or schema_accepts(arguments, minus_tool):
        return None, "execution_check_failed"
    template = query_template(query, value)
    family = call["name"]
    renamed_call = {"name": opaque_name, "arguments": arguments}
    pair = {
        "pair_id": f"f0_{source_id}",
        "source_id": source_id,
        "category": category,
        "tool_family": family,
        "template_id": stable_digest(family, template)[:16],
        "query_template": template,
        "target_parameter": target,
        "gold_target_value": value,
        "constraint_family": constraint_family,
        "historically_exposed": source_id in historical_ids,
        "query": query,
        "plus": {
            "messages": copy.deepcopy(messages),
            "tools": [plus_tool],
            "expected_decision": "call",
            "expected_calls": [renamed_call],
        },
        "minus": {
            "messages": copy.deepcopy(messages),
            "tools": [minus_tool],
            "expected_decision": "decline",
            "expected_calls": [],
        },
        "positive_constraint": positive,
        "negative_constraint": negative,
        "verification": {
            "single_turn": True,
            "single_required_call": True,
            "target_required": True,
            "target_has_no_default": True,
            "query_has_direct_value_evidence": True,
            "single_tool_closed_world": True,
            "plus_gold_schema_pass": True,
            "minus_gold_schema_pass": False,
            "same_tool_name_both_sides": True,
            "same_constraint_class_both_sides": True,
        },
    }
    return pair, "accepted"


def rebuild_as_const(pair: dict[str, Any]) -> dict[str, Any]:
    """把冻结候选改写为训练未见的 const 合同类型。"""
    result = copy.deepcopy(pair)
    value = result["gold_target_value"]
    positive, negative, family = build_constraints(value, result["source_id"], "unseen_const")
    source_tool = remove_constraint(
        result["plus"]["tools"][0], result["target_parameter"], result["positive_constraint"]
    )
    target = result["target_parameter"]
    name = source_tool["function"]["name"]
    # 重新从同一个中性副本构造，确保自然语言描述与机器约束一致。
    plus_tool = apply_constraint(source_tool, name, target, positive)
    minus_tool = apply_constraint(source_tool, name, target, negative)
    arguments = result["plus"]["expected_calls"][0]["arguments"]
    if not schema_accepts(arguments, plus_tool) or schema_accepts(arguments, minus_tool):
        raise RuntimeError(f"{result['pair_id']} 的 const 改写未通过执行检查")
    result["plus"]["tools"] = [plus_tool]
    result["minus"]["tools"] = [minus_tool]
    result["positive_constraint"] = positive
    result["negative_constraint"] = negative
    result["constraint_family"] = family
    return result


def rank_rows(rows: Iterable[dict[str, Any]], seed: int, label: str) -> list[dict[str, Any]]:
    """按固定种子稳定排序，避免依赖原始文件顺序。"""
    return sorted(rows, key=lambda row: stable_digest(seed, label, row["source_id"]))


def choose_ood_families(
    unused: list[dict[str, Any]], target: int, seed: int
) -> tuple[list[dict[str, Any]], set[str]]:
    """选择完整隔离的工具家族作为 OOD-family 测试。"""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unused:
        by_family[row["tool_family"]].append(row)
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    # 优先隔离单样本家族，把有限的重复家族留给严格 ID 开发/测试。
    ordered = sorted(
        by_family,
        key=lambda family: (len(by_family[family]), stable_digest(seed, "ood_family", family)),
    )
    for family in ordered:
        candidates = rank_rows(by_family[family], seed, "ood_family_rows")
        if not candidates:
            continue
        families.add(family)
        remaining = target - len(selected)
        selected.extend(candidates[:remaining])
        if len(selected) >= target:
            break
    if len(selected) != target:
        raise RuntimeError(f"OOD-family 仅选到 {len(selected)}/{target} 对")
    return selected, families


def choose_reserve(
    pool: list[dict[str, Any]], target: int, seed: int, label: str,
    blocked_sources: set[str], blocked_templates: set[str], blocked_families: set[str],
    train_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """选择未暴露的开发或测试任务，并确保同家族仍有独立训练模板。"""
    selected: list[dict[str, Any]] = []
    for row in rank_rows(pool, seed, label):
        if (
            row["source_id"] in blocked_sources
            or row["template_id"] in blocked_templates
            or row["tool_family"] in blocked_families
        ):
            continue
        support = any(
            candidate["tool_family"] == row["tool_family"]
            and candidate["template_id"] != row["template_id"]
            and candidate["source_id"] not in blocked_sources
            for candidate in train_candidates
        )
        if not support:
            continue
        selected.append(row)
        blocked_sources.add(row["source_id"])
        blocked_templates.add(row["template_id"])
        blocked_families.add(row["tool_family"])
        if len(selected) == target:
            return selected
    raise RuntimeError(f"{label} 仅选到 {len(selected)}/{target} 对")


def choose_train(
    pool: list[dict[str, Any]], target: int, seed: int, blocked_sources: set[str],
    blocked_templates: set[str], required_families: set[str],
) -> list[dict[str, Any]]:
    """选择模板去重的训练任务，并覆盖所有 ID 开发/测试家族。"""
    eligible = [
        row for row in pool
        if row["source_id"] not in blocked_sources and row["template_id"] not in blocked_templates
    ]
    selected: list[dict[str, Any]] = []
    used_templates: set[str] = set()
    for family in sorted(required_families):
        options = rank_rows(
            (row for row in eligible if row["tool_family"] == family), seed, f"train_support:{family}"
        )
        if not options:
            raise RuntimeError(f"ID 划分家族 {family} 缺少独立训练模板")
        row = options[0]
        selected.append(row)
        used_templates.add(row["template_id"])
    remainder = sorted(
        eligible,
        key=lambda row: (
            not row["historically_exposed"],
            stable_digest(seed, "train", row["source_id"]),
        ),
    )
    for row in remainder:
        if row["template_id"] in used_templates or row in selected:
            continue
        selected.append(row)
        used_templates.add(row["template_id"])
        if len(selected) == target:
            return selected
    raise RuntimeError(f"训练集仅选到 {len(selected)}/{target} 个独立模板")


def paired_surface_checks(pair: dict[str, Any]) -> dict[str, bool]:
    """检查不解析用户语义的成对表面线索。"""
    plus = pair["plus"]["tools"][0]["function"]
    minus = pair["minus"]["tools"][0]["function"]
    plus_field = plus["parameters"]["properties"][pair["target_parameter"]]
    minus_field = minus["parameters"]["properties"][pair["target_parameter"]]
    plus_json = json.dumps(plus, ensure_ascii=False, sort_keys=True)
    minus_json = json.dumps(minus, ensure_ascii=False, sort_keys=True)
    return {
        "tool_name_differs": plus["name"] != minus["name"],
        "property_order_differs": list(plus["parameters"]["properties"]) != list(minus["parameters"]["properties"]),
        "constraint_key_differs": sorted(set(plus_field) & CONSTRAINT_KEYS) != sorted(set(minus_field) & CONSTRAINT_KEYS),
        "enum_cardinality_differs": len(plus_field.get("enum", [])) != len(minus_field.get("enum", [])),
        "plus_serialized_longer": len(plus_json) > len(minus_json),
        "equal_serialized_length": len(plus_json) == len(minus_json),
    }


def load_candidates(
    data_root: Path, historical_ids: set[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """从四个 BFCL 单轮类别加载并严格筛选候选。"""
    output: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for category in CATEGORIES:
        rows = read_jsonl(data_root / f"BFCL_v4_{category}.json")
        answers = {
            row["id"]: row["ground_truth"]
            for row in read_jsonl(data_root / "possible_answer" / f"BFCL_v4_{category}.json")
        }
        for row in rows:
            source_id = row.get("id")
            if source_id not in answers:
                reasons["missing_answer"] += 1
                continue
            pair, reason = build_candidate(row, category, answers[source_id], historical_ids)
            reasons[reason] += 1
            if pair is not None:
                output.append(pair)
    return output, reasons


def main() -> None:
    """构造、分组冻结并审计 F0 数据。"""
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    gorilla = args.gorilla_root.resolve()
    data_root = gorilla / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"BFCL 数据目录不存在：{data_root}")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已存在的 F0 目录：{output}")

    historical_ids = load_historical_source_ids(root)
    candidates, reasons = load_candidates(data_root, historical_ids)
    if len({row["source_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("候选 source_id 不唯一")
    unused = [row for row in candidates if not row["historically_exposed"]]

    ood_family, ood_families = choose_ood_families(unused, args.test_ood_family, args.seed)
    non_ood = [row for row in candidates if row["tool_family"] not in ood_families]
    unused_non_ood = [row for row in non_ood if not row["historically_exposed"]]
    blocked_sources = {row["source_id"] for row in ood_family}
    blocked_templates = {row["template_id"] for row in ood_family}
    reserve_families: set[str] = set()

    dev = choose_reserve(
        unused_non_ood, args.dev, args.seed, "dev", blocked_sources, blocked_templates,
        reserve_families, non_ood,
    )
    test_id = choose_reserve(
        unused_non_ood, args.test_id, args.seed, "test_id", blocked_sources, blocked_templates,
        reserve_families, non_ood,
    )
    ood_constraint_base = choose_reserve(
        unused_non_ood, args.test_ood_constraint, args.seed, "test_ood_constraint",
        blocked_sources, blocked_templates, reserve_families, non_ood,
    )
    ood_constraint = [rebuild_as_const(row) for row in ood_constraint_base]
    required_families = {
        row["tool_family"] for row in dev + test_id + ood_constraint
    }
    train = choose_train(
        non_ood, args.train, args.seed, blocked_sources, blocked_templates, required_families
    )

    splits = {
        "train": train,
        "dev": dev,
        "test_id": test_id,
        "test_ood_family": ood_family,
        "test_ood_constraint": ood_constraint,
    }
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
            row["surface_shortcut_checks"] = paired_surface_checks(row)

    all_rows = [row for split in splits.values() for row in split]
    source_sets = {name: {row["source_id"] for row in rows} for name, rows in splits.items()}
    template_sets = {name: {row["template_id"] for row in rows} for name, rows in splits.items()}
    split_names = list(splits)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if source_sets[left] & source_sets[right]:
                raise RuntimeError(f"{left} 与 {right} 存在 source_id 泄漏")
            if template_sets[left] & template_sets[right]:
                raise RuntimeError(f"{left} 与 {right} 存在请求模板泄漏")
    train_families = {row["tool_family"] for row in train}
    if train_families & ood_families:
        raise RuntimeError("OOD-family 工具家族泄漏到训练集")
    if any(row["tool_family"] not in train_families for row in dev + test_id + ood_constraint):
        raise RuntimeError("ID/约束 OOD 划分存在训练未覆盖工具家族")

    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "pairs.jsonl"
    write_jsonl(data_path, all_rows)
    for split, rows in splits.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    review = rank_rows(all_rows, args.seed, "manual_review")[:30]
    write_jsonl(output / "manual_review_sample.jsonl", review)

    shortcut_counts: Counter[str] = Counter()
    for row in all_rows:
        for key, value in row["surface_shortcut_checks"].items():
            shortcut_counts[key] += int(value)
    manifest = {
        "version": "ccpd_v11_f0_v2",
        "seed": args.seed,
        "gorilla_commit": (
            __import__("subprocess").check_output(
                ["git", "-C", str(gorilla), "rev-parse", "HEAD"], text=True
            ).strip()
        ),
        "historical_source_ids": len(historical_ids),
        "candidate_count": len(candidates),
        "candidate_unexposed_count": len(unused),
        "candidate_outcomes": dict(sorted(reasons.items())),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "historical_exposure_by_split": {
            name: int(sum(row["historically_exposed"] for row in rows))
            for name, rows in splits.items()
        },
        "constraint_families": dict(Counter(row["constraint_family"] for row in all_rows)),
        "categories": dict(Counter(row["category"] for row in all_rows)),
        "tool_families_by_split": {
            name: len({row["tool_family"] for row in rows}) for name, rows in splits.items()
        },
        "surface_shortcut_counts": dict(shortcut_counts),
        "data_sha256": sha256_file(data_path),
        "guarantees": [
            "同一配对中请求、工具名、工具数量、属性顺序和约束类别保持一致",
            "目标字段必填、无 default，Gold 目标值在请求中有直接文本证据",
            "E+ Gold 通过执行合同，E- Gold 被执行合同拒绝",
            "每侧仅提供同一个闭世界工具，不存在其他工具绕过约束",
            "source_id 与归一化请求模板均不跨划分",
            "test_ood_family 的工具家族与训练集完全隔离",
            "test_ood_constraint 使用训练未出现的 const 约束",
        ],
        "limitations": [
            "自动执行器验证 JSON Schema 子集，不代表真实外部 API 业务语义",
            "训练集优先复用历史 BFCL 来源；开发与测试仅使用本项目此前未暴露 source_id",
            "人工语义审核状态由 manual_review.md 单独记录",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "generation_audit.json").write_text(
        json.dumps(
            {
                "candidate_outcomes": dict(sorted(reasons.items())),
                "surface_shortcut_counts": dict(shortcut_counts),
                "manual_review_status": "pending",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
