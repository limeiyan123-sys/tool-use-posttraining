#!/usr/bin/env python3
"""把冻结 F0 配对转换为 LLaMA-Factory 平衡 SFT 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REJECTION = "无法使用当前工具完成请求。"


def parse_args() -> argparse.Namespace:
    """解析冻结配对和数据输出目录。"""
    parser = argparse.ArgumentParser(description="构建 CCPD-v1.1 F4 平衡 SFT 数据")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} 出现空行")
            rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_tools(tools: list[dict[str, Any]]) -> str:
    """转换 OpenAI wrapper 为 LLaMA-Factory tools 列表。"""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("F4 仅接受标准 function tool")
        normalized.append(tool["function"])
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def make_example(pair: dict[str, Any], side_name: str) -> dict[str, Any]:
    """为可满足侧生成调用监督，为不可满足侧生成规范拒绝监督。"""
    side = pair[side_name]
    messages = side["messages"]
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(f"{pair['pair_id']} 不是单轮用户请求")
    conversations = [{"from": "human", "value": messages[0]["content"]}]
    if side_name == "plus":
        calls = side["expected_calls"]
        if len(calls) != 1:
            raise ValueError(f"{pair['pair_id']} 可满足侧不是单调用")
        conversations.append(
            {
                "from": "function_call",
                "value": json.dumps(calls[0], ensure_ascii=False, sort_keys=True),
            }
        )
    else:
        if side["expected_calls"]:
            raise ValueError(f"{pair['pair_id']} 不可满足侧不应包含 Gold 调用")
        conversations.append({"from": "gpt", "value": REJECTION})
    return {
        "conversations": conversations,
        "tools": normalize_tools(side["tools"]),
        "metadata": {
            "pair_id": pair["pair_id"],
            "side": side_name,
            "split": pair["split"],
            "constraint_family": pair["constraint_family"],
            "tool_family": pair["tool_family"],
        },
    }


def main() -> None:
    """按配对整体划分生成严格 1:1 的训练与开发数据。"""
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖 F4 数据目录：{output}")
    pairs = read_jsonl(args.pairs.resolve())
    selected = {split: [row for row in pairs if row["split"] == split] for split in ("train", "dev")}
    if not selected["train"] or not selected["dev"]:
        raise ValueError("冻结数据缺少 train 或 dev")
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, Path] = {}
    for split, split_pairs in selected.items():
        examples = [make_example(pair, side) for pair in split_pairs for side in ("plus", "minus")]
        path = output / f"{split}.json"
        path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files[split] = path

    dataset_info = {
        "f4_balanced_dev": {
            "file_name": "dev.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
        "f4_balanced_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "tools": "tools"},
        },
    }
    info_path = output / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": "f4_balanced_sft_v1",
        "source_pairs": args.pairs.as_posix(),
        "source_pairs_sha256": sha256_file(args.pairs.resolve()),
        "train_pairs": len(selected["train"]),
        "train_examples": len(selected["train"]) * 2,
        "dev_pairs": len(selected["dev"]),
        "dev_examples": len(selected["dev"]) * 2,
        "call_reject_ratio": "1:1",
        "rejection_target": REJECTION,
        "train_sha256": sha256_file(files["train"]),
        "dev_sha256": sha256_file(files["dev"]),
        "dataset_info_sha256": sha256_file(info_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
