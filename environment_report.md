# Stage 0 环境验证报告

验证时间：2026-10-14 10:35（Asia/Shanghai）

## 结论

新服务器的 CUDA、PyTorch、Transformers、LLaMA-Factory、模型加载、结构化工具调用、QLoRA 两步训练和 adapter 重载链路均已验证通过。当前环境可进入 Stage 1，但本阶段没有进行正式训练，也没有产生可用于论文对比的准确率结果。

## 硬件与系统

| 项目 | 实测值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition × 1 |
| GPU 显存 | 97,887 MiB |
| NVIDIA Driver | 590.44.01 |
| 驱动报告 CUDA | 13.1 |
| PyTorch CUDA Runtime | 12.8 |
| GPU Compute Capability | 12.0 |
| BF16 | 支持，矩阵计算通过 |
| CPU 配额 | 25 vCPU |
| 内存配额 | 120 GiB |
| 系统盘 | 30 GiB，剩余约 6.7 GiB |
| 数据盘 | 150 GiB，Stage 0 后剩余约 147 GiB |

系统未安装 `nvcc`。这不影响当前基于 PyTorch wheel 的推理和训练，但如果后续需要编译 FlashAttention、自定义 CUDA 算子或其他源码扩展，则必须单独安装与 PyTorch CUDA 12.8 兼容的 Toolkit，不能静默退回低性能实现。

## Python 与训练框架

项目使用 `/root/autodl-tmp/envs/tooluse`。该虚拟环境继承服务器 `lf311` 的只读基础依赖，并在项目环境中覆盖冲突版本和新增依赖。

| 包 | 版本 |
|---|---|
| Python | 3.11.6 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.6.0 |
| PEFT | 0.18.1 |
| TRL | 0.24.0 |
| Datasets | 4.0.0 |
| Accelerate | 1.11.0 |
| bitsandbytes | 0.50.2 |
| fsspec | 2025.3.0 |
| Pillow | 11.3.0 |
| LLaMA-Factory | 0.9.5.dev0 |

`pip check` 返回 `No broken requirements found`。完整依赖快照保存在 `reports/stage0_requirements_freeze.txt`，其 SHA256 为 `ee8ee0892ba9ca3ae52fddbbe79b883723e5a41eb6c3cdd69c0641cf931e8f1e`。

LLaMA-Factory 使用服务器已有仓库的已提交 Git 快照，提交为 `dba1838e5f471bee9751364f45a3f18c51efa191`。该来源是服务器已有 fork，而不是本次从官方仓库新下载的快照；正式实验开始前应将框架来源纳入可复现性审计并固定，不应在不同实验中切换提交。

## 模型

主模型位置：`/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`

| 项目 | 结果 |
|---|---|
| 来源 | ModelScope：Qwen/Qwen2.5-1.5B-Instruct |
| 架构 | Qwen2ForCausalLM |
| 权重格式 | Safetensors |
| 权重文件大小 | 3,087,467,144 bytes |
| 权重 SHA256 | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 本地加载 | 通过 |
| 加载精度 | BF16 |

## 推理验证

脚本：`scripts/verify_inference.py`

输入要求模型调用 `weather_api`，并提供必填参数 `city` 与 `date`。模型输出为合法 `<tool_call>` JSON，工具名和参数均与期望严格一致：

```json
{
  "name": "weather_api",
  "arguments": {
    "city": "北京",
    "date": "2026-10-15"
  }
}
```

推理峰值显存约 2.904 GiB。结构化结果保存在 `reports/stage0_inference_result.json`。

该结果仅证明加载、模板和生成链路可用，不代表 Zero-shot benchmark 准确率。

## QLoRA Smoke Test

配置：`configs/stage0_qlora_smoke.yaml`

数据：4 条合成工具调用样本，仅用于链路验证。

| 项目 | 实测值 |
|---|---|
| 量化 | bitsandbytes 4-bit |
| 计算精度 | BF16 |
| LoRA rank / alpha | 8 / 16 |
| LoRA 目标 | q/k/v/o、gate/up/down projection |
| 可训练参数 | 9,232,384 |
| 可训练参数比例 | 0.5945% |
| 优化 step | 2 |
| Step 1 loss | 0.1101201549 |
| Step 2 loss | 0.1050859988 |
| 最终 train loss | 0.1076030768 |
| 训练时长 | 1.233 秒 |
| 结果 | 通过 |

adapter 权重位于 `checkpoints/stage0_qlora_smoke/adapter_model.safetensors`，大小为 36,981,072 bytes，SHA256 为 `bc494c04a5445af58d06c72a975606c08be118da53d224e30eb7b24d0207f1f3`。

完整控制台日志保存在 `logs/stage0_qlora_smoke_console.log`。该测试没有验证集，因此产生“无 eval loss/accuracy 可绘制”的 warning，符合预期，不能解读为模型效果。

## Adapter 重载验证

脚本：`scripts/verify_adapter.py`

基础模型以 NF4 四比特双重量化重新加载，随后加载 Stage 0 adapter。模型成功生成 `calculator_api` 工具调用，峰值显存约 1.157 GiB。结构化结果保存在 `reports/stage0_adapter_reload_result.json`。

## Warning 与限制

1. LLaMA-Factory 提示量化训练建议启用 `upcast_layernorm`。正式 QLoRA 配置应开启该选项，并单独验证数值稳定性。
2. 当前 smoke 配置使用的 `warmup_ratio` 和 `logging_dir` 被 Transformers 标记为未来弃用。正式配置应改用 `warmup_steps`，并通过环境变量设置 TensorBoard 日志目录。
3. 训练时 tokenizer 的 PAD/BOS/EOS 配置被 Trainer 对齐。正式实验必须固定 tokenizer 与模板版本，并在日志中保留对齐信息。
4. 推理时 Transformers 提示模型 generation config 中的 sampling 参数在贪心生成下被忽略。当前测试明确使用确定性生成，该提示不影响结论。
5. LLaMA-Factory fork 中跟踪了部分 `__pycache__` 文件，导入后会产生字节码状态变化；源代码提交号未变化。正式项目仓库不应跟踪字节码文件。
6. Stage 0 合成数据不得进入正式训练集、验证集或测试集，也不得计入后续准确率表格。

## Stage 0 判定

- CUDA 与 BF16：通过。
- PyTorch 与依赖一致性：通过。
- LLaMA-Factory CLI：通过。
- Qwen2.5-1.5B-Instruct 本地加载：通过。
- 工具调用推理：通过。
- QLoRA 训练链路：通过。
- adapter 保存与重载：通过。
- 正式训练：未开始。
