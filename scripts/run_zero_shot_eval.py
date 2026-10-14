#!/usr/bin/env python3
"""在冻结评测集上运行零样本工具调用推理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行零样本工具调用评测")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并拒绝空行。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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


def load_and_verify_datasets(frozen_dir: Path) -> list[dict[str, Any]]:
    """读取冻结清单并校验数据文件完整性。"""
    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for filename, metadata in manifest["datasets"].items():
        path = frozen_dir / filename
        actual_hash = sha256_file(path)
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(f"冻结数据哈希不匹配：{path}")
        current = read_jsonl(path)
        if len(current) != metadata["samples"]:
            raise RuntimeError(f"冻结数据样本数不匹配：{path}")
        rows.extend(current)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("冻结评测集中存在重复 ID")
    return sorted(rows, key=lambda row: row["id"])


def prepare_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """加入固定系统指令，同时保持原始用户消息不变。"""
    system_message = {
        "role": "system",
        "content": (
            "你是严谨的工具调用助手。仅在工具与用户请求相关时调用工具；"
            "需要调用时只输出模板要求的结构化工具调用，不要解释或猜测缺失参数。"
        ),
    }
    if messages and messages[0].get("role") == "system":
        raise ValueError("评测样本不应自行携带 system 消息")
    return [system_message, *messages]


def main() -> None:
    """加载模型并对每个冻结样本进行确定性生成。"""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    project_root = Path(config["project_root"]).resolve()
    frozen_dir = Path(
        config.get("frozen_dir", project_root / "data" / "evaluation" / "frozen_v1")
    ).resolve()
    model_path = Path(config["model_path"]).resolve()
    output_path = Path(config["output_path"]).resolve()
    if output_path.exists():
        raise FileExistsError(f"预测文件已存在，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(int(config["seed"]))
    samples = load_and_verify_datasets(frozen_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        raise FileExistsError(f"发现上次未完成的临时文件，请先人工检查：{temporary_path}")

    started = time.perf_counter()
    with temporary_path.open("x", encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(samples, start=1):
            messages = prepare_messages(sample["messages"])
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=sample["tools"],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            sample_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(config["max_new_tokens"]),
                    do_sample=False,
                    use_cache=True,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            record = {
                "id": sample["id"],
                "response": response,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(new_tokens.shape[0]),
                "latency_seconds": round(time.perf_counter() - sample_started, 6),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index == 1 or index % 10 == 0 or index == len(samples):
                print(f"已完成 {index}/{len(samples)}")

    temporary_path.replace(output_path)
    metadata = {
        "model_path": str(model_path),
        "frozen_dir": str(frozen_dir),
        "frozen_manifest_sha256": sha256_file(frozen_dir / "manifest.json"),
        "model_class": model.__class__.__name__,
        "dtype": str(model.dtype),
        "seed": int(config["seed"]),
        "decoding": {"do_sample": False, "max_new_tokens": int(config["max_new_tokens"])},
        "samples": len(samples),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "prediction_sha256": sha256_file(output_path),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
