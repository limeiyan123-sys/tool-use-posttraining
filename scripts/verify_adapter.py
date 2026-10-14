"""验证 Stage 0 生成的 QLoRA adapter 可以被重新加载和推理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    """解析基础模型、adapter 和输出文件路径。"""
    parser = argparse.ArgumentParser(description="验证 QLoRA adapter 重载链路")
    parser.add_argument("--model-path", required=True, help="本地基础模型目录")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter 目录")
    parser.add_argument("--output-path", required=True, help="JSON 结果文件")
    return parser.parse_args()


def main() -> None:
    """以 NF4 四比特方式加载基础模型和 adapter，并完成一次生成。"""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝退回 CPU 验证")

    model_path = Path(args.model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    output_path = Path(args.output_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"基础模型目录不存在：{model_path}")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter 权重不存在：{adapter_path}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "计算 17 乘以 29。"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator_api",
                    "description": "计算数学表达式",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
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
    result = {
        "base_model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "adapter_loaded": True,
        "quantization": "bitsandbytes-nf4-double-quant",
        "response": response,
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
