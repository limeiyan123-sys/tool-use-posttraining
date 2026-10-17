# Checkpoint 说明

本目录保存八组正式实验的最终 LoRA adapter，并通过 Git LFS 版本化。仓库不包含 Qwen2.5-1.5B-Instruct 基座权重，也不包含各训练过程的 `checkpoint-*` 优化器快照；使用前需要自行获取基座模型。

每个实验目录保留以下恢复材料：

- `adapter_model.safetensors` 与 `adapter_config.json`：LoRA 权重和结构配置；
- `tokenizer.json`、`tokenizer_config.json` 与 `chat_template.jinja`：训练时使用的分词和对话模板；
- `training_args.bin`、`trainer_state.json`、`trainer_log.jsonl`：训练参数、状态和逐步日志；
- `train_results.json`、`eval_results.json` 与曲线图片：结果摘要和训练诊断。

这些文件足以加载最终 adapter 继续微调或执行评测。若要从某个中间 step 连同优化器状态精确续训，仍需实验服务器上的对应 `checkpoint-*` 目录。

## 权重清单

| 实验目录 | 大小 | SHA-256 |
| --- | ---: | --- |
| `stage1_lora_baseline` | 70.49 MiB | `dcf40498b1d5eb2459ae924438af362a1bd5b3e6851a9499698652ac40431b36` |
| `stage3_failure_driven` | 70.49 MiB | `e2ee428936015e57d2eb9919c76266dd80fee68a5fa9f8d0f81be5c67f6ffda4` |
| `stage3_schema_only` | 70.49 MiB | `fb818ba63d221c5d78f4f8db135d70082a1a645245de7d4b5794328e31b4264d` |
| `stage4_counterfactual_paired` | 70.49 MiB | `2d12583264bf32f4038ddd886a5e6139f7b57f02b9b421030a4825a6b10af6c5` |
| `stage4_failure_weighted` | 70.49 MiB | `8e173e08027b248b025ccab909b5fdb5fa0b0407a398ab324685e0d8b3ed561c` |
| `stage4_world_randomization` | 70.49 MiB | `0719f21153c27903d5a16693df2354d76d5989a9e01a749b120bae9de5be7182` |
| `stage5_fctw` | 70.49 MiB | `e408b8230d945288b54bbb07294ebe301bfe4cc3f1c8b666652e7f52b24013d2` |
| `ccpd_v11_f4_balanced_sft` | 35.27 MiB | `c52125a7a1fdbdc8129cad8bb610975396c5060bacc5ab284f21031666b8a6e4` |

## 获取与校验

克隆仓库时应安装 Git LFS：

```bash
git lfs install
git clone https://github.com/limeiyan123-sys/tool-use-posttraining.git
cd tool-use-posttraining
git lfs pull
```

加载方式、对应训练配置和实验结论见根目录 `README.md`、`configs/` 与 `reports/`。

