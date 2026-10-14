"""验证模型加载、工具调用模板和确定性生成链路。"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    """解析模型路径和结果文件路径。"""
    parser = argparse.ArgumentParser(description="验证 Qwen 工具调用推理链路")
    parser.add_argument("--model-path", required=True, help="本地模型目录")
    parser.add_argument("--output-path", required=True, help="JSON 结果文件")
    return parser.parse_args()


def build_tools() -> list[dict]:
    """构造包含必填参数的最小天气工具 Schema。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "weather_api",
                "description": "查询指定城市和日期的天气。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "date": {"type": "string", "description": "ISO 8601 日期"},
                    },
                    "required": ["city", "date"],
                },
            },
        }
    ]


def parse_tool_call(text: str) -> tuple[bool, dict | None, str | None]:
    """严格解析模型输出中的首个工具调用，不对错误输出做猜测性修复。"""
    match = TOOL_CALL_PATTERN.search(text)
    if match is None:
        return False, None, "未找到 <tool_call> 标签"

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return False, None, f"工具调用不是合法 JSON：{exc}"

    expected = {
        "name": "weather_api",
        "arguments": {"city": "北京", "date": "2026-10-15"},
    }
    if payload != expected:
        return False, payload, f"工具调用与期望不一致：{payload!r}"
    return True, payload, None


def main() -> None:
    """加载模型并执行一次确定性的工具调用推理。"""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝退回 CPU 推理")

    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "你是工具调用助手。需要外部信息时只输出结构化工具调用。",
        },
        {
            "role": "user",
            "content": "请查询北京在 2026-10-15 的天气。",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=build_tools(),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    )
    is_valid, tool_call, parse_error = parse_tool_call(response)
    if parse_error is not None:
        warnings.warn(parse_error, RuntimeWarning, stacklevel=1)

    result = {
        "model_path": str(model_path),
        "model_class": model.__class__.__name__,
        "dtype": str(model.dtype),
        "gpu": torch.cuda.get_device_name(0),
        "prompt": prompt,
        "response": response,
        "tool_call_valid": is_valid,
        "tool_call": tool_call,
        "parse_error": parse_error,
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
