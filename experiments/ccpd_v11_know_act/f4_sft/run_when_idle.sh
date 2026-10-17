#!/usr/bin/env bash
set -euo pipefail

# 仅在共享 GPU 1 连续空闲时启动，避免与其他研究任务争抢显存和算力。
PROJECT_ROOT="/home/limeiyan/tool-use-posttraining"
CONFIG="$PROJECT_ROOT/experiments/ccpd_v11_know_act/f4_sft/f4_balanced_lora.yaml"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/ccpd_v11_f4_balanced_sft"
STATUS_DIR="$PROJECT_ROOT/experiments/ccpd_v11_know_act/f4_sft/results"
POLL_SECONDS=60
MAX_POLLS=120
MIN_FREE_MIB=18000
MAX_UTIL=40
REQUIRED_STREAK=3

mkdir -p "$STATUS_DIR"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "拒绝覆盖已有 F4 输出：$OUTPUT_DIR" >&2
  exit 2
fi

streak=0
for ((attempt=1; attempt<=MAX_POLLS; attempt++)); do
  IFS=',' read -r used free util < <(
    nvidia-smi --id=1 --query-gpu=memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' '
  )
  printf '%s attempt=%d used=%s free=%s util=%s streak=%d\n' \
    "$(date --iso-8601=seconds)" "$attempt" "$used" "$free" "$util" "$streak"
  if (( free >= MIN_FREE_MIB && util <= MAX_UTIL )); then
    streak=$((streak + 1))
  else
    streak=0
  fi
  if (( streak >= REQUIRED_STREAK )); then
    cd "$PROJECT_ROOT"
    export CUDA_VISIBLE_DEVICES=1
    export TOKENIZERS_PARALLELISM=false
    echo "$(date --iso-8601=seconds) F4 资源门控通过，开始训练"
    /opt/anaconda3/bin/llamafactory-cli train "$CONFIG"

    # 训练成功后立即使用冻结协议完成原生行为和隐状态评测。
    PYTHON="/opt/anaconda3/envs/llama-factory-env/bin/python"
    PAIRS="$PROJECT_ROOT/experiments/ccpd_v11_know_act/f0_data/frozen_v2/pairs.jsonl"
    MODEL="/home/limeiyan/models/Qwen2.5-1.5B-Instruct"
    PREDICTIONS="$STATUS_DIR/f4_predictions.jsonl"
    "$PYTHON" experiments/ccpd_v11_know_act/f1_prompt/run_f1_prompts.py \
      --pairs "$PAIRS" --model-path "$MODEL" --adapter-path "$OUTPUT_DIR" \
      --output "$PREDICTIONS" --model-name f4_balanced_sft \
      --conditions A_native --gpu-memory-fraction 0.10
    "$PYTHON" experiments/ccpd_v11_know_act/f1_prompt/analyze_f1.py \
      --project-root "$PROJECT_ROOT" --pairs "$PAIRS" --predictions "$PREDICTIONS" \
      --output "$STATUS_DIR/f4_summary.json"
    "$PYTHON" experiments/ccpd_v11_know_act/f2_probe/extract_native_hidden.py \
      --pairs "$PAIRS" --model-path "$MODEL" --adapter-path "$OUTPUT_DIR" \
      --output-dir "$STATUS_DIR/f4_hidden" --model-name f4_balanced_sft \
      --gpu-memory-fraction 0.10
    "$PYTHON" experiments/ccpd_v11_know_act/f2_probe/analyze_probes.py \
      --pairs "$PAIRS" --hidden-dir "$STATUS_DIR/f4_hidden" \
      --output "$STATUS_DIR/f4_probe_summary.json"
    date --iso-8601=seconds > "$STATUS_DIR/F4_COMPLETE"
    echo "$(date --iso-8601=seconds) F4 训练与冻结评测全部完成"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

echo "$(date --iso-8601=seconds) 两小时内未获得安全资源窗口，F4 未启动" >&2
exit 3
