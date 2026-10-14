#!/usr/bin/env python3
"""清洗 ToolACE，并构建普通 LoRA SFT 的训练集与开发集。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


OFFICIAL_SHA256 = "ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e"
TYPE_MAPPING = {"dict": "object", "int": "integer", "float": "number", "bool": "boolean"}
VALID_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="准备 ToolACE SFT 数据")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--max-tool-rows", type=int, default=1200)
    parser.add_argument("--max-no-tool-rows", type=int, default=300)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, value: str) -> str:
    """生成与输入顺序无关的稳定排序键。"""
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """归一化文本，仅用于去重和污染检查。"""
    return re.sub(r"[^\w]+", "", text.casefold(), flags=re.UNICODE)


def ngrams(text: str, size: int = 4) -> set[str]:
    """生成字符 n-gram 集合。"""
    normalized = normalize_text(text)
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """计算两个集合的 Jaccard 相似度。"""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def normalize_type(value: Any) -> str | list[str]:
    """将 ToolACE 类型映射为 JSON Schema 类型，非法类型直接抛错。"""
    if isinstance(value, str):
        normalized = TYPE_MAPPING.get(value, value)
        if normalized not in VALID_TYPES:
            raise ValueError(f"不支持的 Schema 类型：{value!r}")
        return normalized
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        normalized_list = [TYPE_MAPPING.get(item, item) for item in value]
        if any(item not in VALID_TYPES for item in normalized_list):
            raise ValueError(f"不支持的联合 Schema 类型：{value!r}")
        return normalized_list
    raise TypeError(f"Schema type 必须是字符串或字符串列表：{value!r}")


def normalize_schema(node: Any) -> Any:
    """递归规范化 JSON Schema，不对结构错误做猜测修复。"""
    if isinstance(node, list):
        return [normalize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    output: dict[str, Any] = {}
    for key, value in node.items():
        if value is None:
            continue
        if key == "type":
            output[key] = normalize_type(value)
        else:
            output[key] = normalize_schema(value)
    if output.get("type") == "object":
        if not isinstance(output.get("properties"), dict):
            raise TypeError("object Schema 缺少 properties 对象")
        required = output.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise TypeError("required 必须是字符串列表")
        if any(item not in output["properties"] for item in required):
            raise ValueError("required 引用了未声明属性")
        output["required"] = required
        output["additionalProperties"] = False
    if output.get("type") == "array" and "items" not in output:
        raise ValueError("array Schema 缺少 items")
    return output


def extract_tools(system: str) -> list[dict[str, Any]]:
    """从 ToolACE 系统提示中严格提取工具 JSON 列表。"""
    start = system.find("[")
    if start < 0:
        raise ValueError("系统提示中没有工具列表")
    raw_tools, _ = json.JSONDecoder().raw_decode(system[start:])
    if not isinstance(raw_tools, list):
        raise TypeError("工具定义不是列表")

    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            raise TypeError("工具定义不是对象")
        name = raw_tool.get("name")
        description = raw_tool.get("description")
        parameters = raw_tool.get("parameters")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("工具名缺失")
        if name in names:
            raise ValueError(f"工具名重复：{name}")
        if not isinstance(description, str):
            raise TypeError(f"工具 {name} 缺少字符串描述")
        normalized_parameters = normalize_schema(parameters)
        if normalized_parameters.get("type") != "object":
            raise ValueError(f"工具 {name} 的顶层参数类型不是 object")
        tools.append({"name": name, "description": description, "parameters": normalized_parameters})
        names.add(name)
    return tools


def split_top_level_calls(content: str) -> list[str]:
    """按顶层逗号切分并行调用，同时保留嵌套结构内的逗号。"""
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(content):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("调用括号不匹配")
        elif char == "," and depth == 0:
            parts.append(content[start:index].strip())
            start = index + 1
    if quote is not None or depth != 0:
        raise ValueError("调用中存在未闭合引号或括号")
    parts.append(content[start:].strip())
    return [part for part in parts if part]


def parse_call_arguments(text: str) -> dict[str, Any]:
    """使用 Python AST 解析关键字参数，仅接受字面量。"""
    expression = ast.parse(f"placeholder({text})", mode="eval").body
    if not isinstance(expression, ast.Call) or expression.args:
        raise ValueError("只支持关键字参数调用")
    arguments: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("不支持 **kwargs")
        if keyword.arg in arguments:
            raise ValueError(f"参数重复：{keyword.arg}")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return arguments


def parse_tool_calls(value: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 ToolACE 的方括号函数语法转换为结构化调用。"""
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("工具调用必须由方括号包围")
    inner = stripped[1:-1].strip()
    if not inner:
        raise ValueError("工具调用列表为空")
    names = sorted((tool["name"] for tool in tools), key=len, reverse=True)
    calls: list[dict[str, Any]] = []
    for part in split_top_level_calls(inner):
        name = next((candidate for candidate in names if part.startswith(candidate + "(")), None)
        if name is None or not part.endswith(")"):
            raise ValueError(f"调用了未声明工具或格式非法：{part[:100]}")
        arguments = parse_call_arguments(part[len(name) + 1 : -1])
        calls.append({"name": name, "arguments": arguments})
    return calls


def validate_arguments(arguments: dict[str, Any], parameters: dict[str, Any]) -> None:
    """检查调用参数名是否由 Schema 声明且必填参数齐全。"""
    properties = parameters["properties"]
    required = parameters.get("required", [])
    unknown = set(arguments) - set(properties)
    missing = set(required) - set(arguments)
    if unknown:
        raise ValueError(f"出现未声明参数：{sorted(unknown)}")
    if missing:
        raise ValueError(f"缺少必填参数：{sorted(missing)}")


def convert_conversations(
    conversations: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[list[dict[str, str]], bool, list[str]]:
    """转换完整对话，并返回是否含工具调用及全部用户文本。"""
    converted: list[dict[str, str]] = []
    has_tool_call = False
    user_texts: list[str] = []
    tool_map = {tool["name"]: tool["parameters"] for tool in tools}

    for turn in conversations:
        role = turn.get("from")
        value = turn.get("value")
        if not isinstance(value, str):
            raise TypeError("对话 value 不是字符串")
        if role == "user":
            converted.append({"from": "human", "value": value.strip()})
            user_texts.append(value.strip())
        elif role == "tool":
            converted.append({"from": "observation", "value": value.strip()})
        elif role == "assistant":
            stripped = value.strip()
            if stripped.startswith("["):
                calls = parse_tool_calls(stripped, tools)
                for call in calls:
                    validate_arguments(call["arguments"], tool_map[call["name"]])
                payload: dict[str, Any] | list[dict[str, Any]] = calls[0] if len(calls) == 1 else calls
                converted.append(
                    {"from": "function_call", "value": json.dumps(payload, ensure_ascii=False, sort_keys=True)}
                )
                has_tool_call = True
            else:
                converted.append({"from": "gpt", "value": stripped})
        else:
            raise ValueError(f"未知角色：{role!r}")

    if not converted or len(converted) % 2 != 0:
        raise ValueError("对话长度必须是非零偶数")
    input_roles = {"human", "observation"}
    output_roles = {"gpt", "function_call"}
    for index, turn in enumerate(converted):
        expected_roles = input_roles if index % 2 == 0 else output_roles
        if turn["from"] not in expected_roles:
            raise ValueError(f"第 {index + 1} 条消息角色顺序非法：{turn['from']}")
    return converted, has_tool_call, user_texts


def load_frozen_signatures(project_root: Path) -> tuple[set[str], set[str], list[tuple[str, set[str]]]]:
    """读取冻结测试集的工具名、工具定义和问题 n-gram。"""
    frozen_dir = project_root / "data" / "evaluation" / "frozen_v1"
    tool_names: set[str] = set()
    tool_signatures: set[str] = set()
    questions: list[tuple[str, set[str]]] = []
    for path in sorted(frozen_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for tool_wrapper in row["tools"]:
                function = tool_wrapper["function"]
                tool_names.add(function["name"])
                tool_signatures.add(json.dumps(function, ensure_ascii=False, sort_keys=True))
            for message in row["messages"]:
                if message["role"] == "user":
                    questions.append((normalize_text(message["content"]), ngrams(message["content"])))
    return tool_names, tool_signatures, questions


def contamination_reason(
    tools: list[dict[str, Any]],
    user_texts: list[str],
    frozen_tool_names: set[str],
    frozen_tool_signatures: set[str],
    frozen_questions: list[tuple[str, set[str]]],
) -> str | None:
    """检查工具名、完整 Schema、精确问题和高相似问题污染。"""
    if any(tool["name"] in frozen_tool_names for tool in tools):
        return "tool_name_overlap"
    if any(json.dumps(tool, ensure_ascii=False, sort_keys=True) in frozen_tool_signatures for tool in tools):
        return "tool_schema_overlap"
    for text in user_texts:
        normalized = normalize_text(text)
        grams = ngrams(text)
        for frozen_normalized, frozen_grams in frozen_questions:
            if normalized == frozen_normalized:
                return "question_exact_overlap"
            if len(normalized) >= 20 and jaccard(grams, frozen_grams) >= 0.8:
                return "question_near_overlap"
    return None


def select_rows(rows: list[dict[str, Any]], size: int, seed: int, salt: str) -> list[dict[str, Any]]:
    """按内容哈希稳定选择指定数量样本。"""
    ranked = sorted(
        rows,
        key=lambda row: stable_rank(seed, salt + json.dumps(row, ensure_ascii=False, sort_keys=True)),
    )
    return ranked[: min(size, len(ranked))]


def split_train_dev(rows: list[dict[str, Any]], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """按内容哈希稳定划分训练集和开发集。"""
    ranked = sorted(rows, key=lambda row: stable_rank(seed, json.dumps(row, ensure_ascii=False, sort_keys=True)))
    dev_size = max(1, round(len(ranked) * dev_ratio))
    return ranked[dev_size:], ranked[:dev_size]


def main() -> None:
    """执行转换、质量过滤、污染检查、抽样和冻结。"""
    args = parse_args()
    project_root = args.project_root.resolve()
    source = args.source.resolve()
    source_hash = sha256_file(source)
    if source_hash != OFFICIAL_SHA256:
        raise RuntimeError(f"ToolACE 源文件哈希错误：{source_hash}")
    raw_rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise TypeError("ToolACE 顶层结构不是列表")

    frozen = load_frozen_signatures(project_root)
    rejection_counts: Counter[str] = Counter()
    tool_rows: list[dict[str, Any]] = []
    no_tool_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_row in raw_rows:
        try:
            tools = extract_tools(raw_row["system"])
            conversations, has_tool_call, user_texts = convert_conversations(raw_row["conversations"], tools)
            reason = contamination_reason(tools, user_texts, *frozen)
            if reason is not None:
                rejection_counts[reason] += 1
                continue
            converted = {
                "conversations": conversations,
                "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
            }
            fingerprint = hashlib.sha256(
                json.dumps(converted, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if fingerprint in seen:
                rejection_counts["exact_duplicate"] += 1
                continue
            seen.add(fingerprint)
            (tool_rows if has_tool_call else no_tool_rows).append(converted)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, SyntaxError) as error:
            rejection_counts[f"invalid::{type(error).__name__}"] += 1

    selected_tool = select_rows(tool_rows, args.max_tool_rows, args.seed, "tool")
    selected_no_tool = select_rows(no_tool_rows, args.max_no_tool_rows, args.seed, "no_tool")
    tool_train, tool_dev = split_train_dev(selected_tool, args.dev_ratio, args.seed)
    no_tool_train, no_tool_dev = split_train_dev(selected_no_tool, args.dev_ratio, args.seed + 1)
    train_rows = select_rows(tool_train + no_tool_train, len(tool_train + no_tool_train), args.seed + 2, "train")
    dev_rows = select_rows(tool_dev + no_tool_dev, len(tool_dev + no_tool_dev), args.seed + 3, "dev")

    output_dir = project_root / "data" / "sft" / "baseline_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.json"
    dev_path = output_dir / "dev.json"
    manifest_path = output_dir / "manifest.json"
    report_path = project_root / "reports" / "stage1" / "data_quality_report.md"
    for path in (train_path, dev_path, manifest_path, report_path):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")

    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dev_path.write_text(json.dumps(dev_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": "toolace_baseline_v1",
        "source": "Team-ACE/ToolACE",
        "source_url": "https://huggingface.co/datasets/Team-ACE/ToolACE",
        "download_mirror": "https://hf-mirror.com/datasets/Team-ACE/ToolACE/resolve/main/data.json",
        "license": "Apache-2.0",
        "source_sha256": source_hash,
        "seed": args.seed,
        "raw_rows": len(raw_rows),
        "valid_tool_rows": len(tool_rows),
        "valid_no_tool_rows": len(no_tool_rows),
        "selected_tool_rows": len(selected_tool),
        "selected_no_tool_rows": len(selected_no_tool),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Stage 1 SFT 数据质量报告

## 数据来源

- 数据集：`Team-ACE/ToolACE`
- 许可证：Apache-2.0
- 官方文件 SHA-256：`{source_hash}`
- 下载方式：Hugging Face 镜像；哈希已与官方 Git LFS 对象核对一致

## 处理结果

| 项目 | 数量 |
|---|---:|
| 原始对话 | {len(raw_rows)} |
| 通过质量与污染检查的工具调用对话 | {len(tool_rows)} |
| 通过质量与污染检查的 no-tool 对话 | {len(no_tool_rows)} |
| 入选工具调用对话 | {len(selected_tool)} |
| 入选 no-tool 对话 | {len(selected_no_tool)} |
| 训练集 | {len(train_rows)} |
| 开发集 | {len(dev_rows)} |

## 剔除统计

```json
{json.dumps(dict(sorted(rejection_counts.items())), ensure_ascii=False, indent=2)}
```

## 质量策略

- 工具 Schema 必须是可验证 JSON Schema 子集；损坏结构不修复。
- 函数调用只接受已声明工具、关键字参数和 Python 字面量。
- 必填参数缺失、额外参数、角色顺序异常均剔除。
- 与冻结测试集重合的工具名、完整工具定义、精确问题和高相似问题均剔除。
- 训练/开发集按稳定内容哈希划分，不依赖源文件顺序。

## 限制

- ToolACE 是合成数据，领域和表达分布不等同于真实用户流量。
- 本基线采用有限子集控制单卡训练成本，不代表全量 ToolACE 训练效果。
- 普通 LoRA 数据未专门增强 Schema 重命名或困难负例；这些留给 Proposed Method。
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
