#!/usr/bin/env python3
"""按顺序运行 Stage 9 推理，记录资源状态和不可覆盖的运行清单。"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from stage9_schema_binding import sha256_file


def main() -> None:
    """检查单卡空闲显存并执行已预注册的直接调用或免训练对照。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["direct", "baselines", "field"])
    parser.add_argument("--fctw-adapter", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / ("data/evaluation/stage9_field_probe_v1" if args.phase == "field" else "data/evaluation/stage9_schema_binding_v2")
    output = root / "reports/stage9_schema_binding/predictions"
    logs = root / "logs/stage9"
    script = root / "scripts/stage9_schema_binding.py"
    models = {"base": None, "fdsrc": "stage3_failure_driven", "fctw": "stage5_fctw"}
    strategies = ["explicit_mapping", "description_alignment"] if args.phase == "baselines" else ["direct"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    record_path = logs / f"{args.phase}_suite.jsonl"
    with record_path.open("x", encoding="utf-8") as records:
        for model, adapter in models.items():
            for strategy in strategies:
                tag = f"{model}_field_probe_v1" if args.phase == "field" else f"{model}_{strategy}_v2"
                resource = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                gpu0 = next(line for line in resource.splitlines() if line.split(",")[0].strip() == "0")
                if int(gpu0.split(",")[1]) < 8192:
                    raise RuntimeError("GPU 0 空闲显存不足 8GB；保留日志并停止本队列，不影响他人进程")
                command = [sys.executable, str(script), "run", "--dataset-dir", str(data), "--model-path", "/home/limeiyan/models/Qwen2.5-1.5B-Instruct", "--output-path", str(output / f"{tag}.jsonl"), "--strategy", strategy]
                if args.phase == "baselines":
                    command.extend(["--conditions", "B", "D"])
                if args.phase == "field":
                    command.extend(["--dataset-name", "field_probe.jsonl", "--max-new-tokens", "32"])
                if adapter:
                    adapter_path = args.fctw_adapter if model == "fctw" and args.fctw_adapter else root / "checkpoints" / adapter
                    command.extend(["--adapter-path", str(adapter_path)])
                record = {"run": tag, "command": command, "gpu_before": resource, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "script_sha256": sha256_file(script), "started_unix": time.time()}
                if adapter:
                    record["adapter_sha256"] = sha256_file(adapter_path / "adapter_model.safetensors")
                print(f"开始 {tag}", flush=True)
                with (logs / f"{tag}.log").open("x", encoding="utf-8") as handle:
                    completed = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
                record["returncode"] = completed.returncode
                record["elapsed_seconds"] = time.time() - record["started_unix"]
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.flush()
                if completed.returncode:
                    raise RuntimeError(f"{tag} 失败，详见对应日志；不隐式更换设备或协议")
                print(f"完成 {tag}，耗时 {record['elapsed_seconds']:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()
