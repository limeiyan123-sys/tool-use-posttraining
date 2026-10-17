#!/usr/bin/env python3
"""在原生工具调用生成起点提取少量代表层隐藏状态。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


NATIVE_SYSTEM = (
    "你是严谨的工具调用助手。仅当候选工具的实际功能和参数约束能够完整满足请求时调用；"
    "否则不要调用工具。需要调用时只输出结构化工具调用。"
)


def parse_args() -> argparse.Namespace:
    """解析模型、冻结配对、批大小与输出参数。"""
    parser = argparse.ArgumentParser(description="提取 CCPD-v1.1 F2 原生隐藏状态")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="base")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.10)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL。"""
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


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_indices(num_layers: int) -> dict[str, int]:
    """预注册 embedding、中层、后层和最后层索引。"""
    if num_layers < 4:
        raise ValueError(f"模型层数过少：{num_layers}")
    return {
        "embedding": 0,
        "middle": max(1, num_layers // 2),
        "late": max(1, (num_layers * 3) // 4),
        "last": num_layers,
    }


def expand_samples(pairs: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """按配对顺序展开两侧并构造原生 Chat Template。"""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for side_name, label in (("plus", 1), ("minus", 0)):
            side = pair[side_name]
            messages = [{"role": "system", "content": NATIVE_SYSTEM}, *side["messages"]]
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=side["tools"],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "source_id": pair["source_id"],
                    "side": side_name,
                    "label": label,
                    "split": pair["split"],
                    "tool_family": pair["tool_family"],
                    "constraint_family": pair["constraint_family"],
                    "prompt": prompt,
                }
            )
    return rows


def main() -> None:
    """批量前向并保存最终上下文位置的代表层向量。"""
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖 F2 隐状态目录：{output}")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须为正数")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path.resolve(), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path.resolve(),
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter_path.resolve(), local_files_only=True)
        if args.adapter_path
        else base
    )
    model.eval()
    num_layers = int(base.config.num_hidden_layers)
    selected_layers = layer_indices(num_layers)
    pairs = read_jsonl(args.pairs.resolve())
    samples = expand_samples(pairs, tokenizer)
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in selected_layers}
    key_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        positions = encoded["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if outputs.hidden_states is None or len(outputs.hidden_states) != num_layers + 1:
            raise RuntimeError("模型未返回预期数量的隐藏状态")
        for name, layer_index in selected_layers.items():
            hidden = outputs.hidden_states[layer_index]
            selected = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), positions
            ].float().cpu().numpy()
            vectors[name].append(selected)
        for offset, row in enumerate(batch):
            key_rows.append(
                {
                    key: value for key, value in row.items() if key != "prompt"
                }
                | {
                    "index": start + offset,
                    "prompt_tokens": int(encoded["attention_mask"][offset].sum()),
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                }
            )
        completed = min(start + len(batch), len(samples))
        if completed == len(batch) or completed % 40 == 0 or completed == len(samples):
            print(f"{args.model_name}: 已提取 {completed}/{len(samples)}", flush=True)
        del outputs, encoded

    output.mkdir(parents=True, exist_ok=False)
    arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in vectors.items()}
    if any(array.shape[0] != len(samples) for array in arrays.values()):
        raise RuntimeError("隐藏状态样本数量不一致")
    np.savez_compressed(output / "hidden_states.npz", **arrays)
    with (output / "keys.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in key_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "model_name": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "pairs": len(pairs),
        "samples": len(samples),
        "num_hidden_layers": num_layers,
        "selected_hidden_state_indices": selected_layers,
        "hidden_size": int(next(iter(arrays.values())).shape[1]),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "native_system": NATIVE_SYSTEM,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0.0
        ),
        "pairs_sha256": sha256_file(args.pairs.resolve()),
        "keys_sha256": sha256_file(output / "keys.jsonl"),
        "arrays_sha256": sha256_file(output / "hidden_states.npz"),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
