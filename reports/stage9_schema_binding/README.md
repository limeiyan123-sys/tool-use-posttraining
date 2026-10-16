# Stage 9：参数标识与字段顺序诊断

本目录归档三个既有模型上的推理实验，不含新训练权重。主结论见 [中文研究报告](stage9_report.md)，固定规则与故障保留见 [实验协议](protocol_and_corrections.md)。

## 产物索引

- `../../configs/stage9_schema_binding.yaml`：模型、解码与统计设置。
- `../../data/evaluation/stage9_schema_binding_v2/`：6 条 pilot、24 条正式基础任务、四条件数据、渲染提示、分词记录与 SHA-256。
- `../../data/evaluation/stage9_field_probe_v1/`：按配对错误选择的字段识别诊断集，不是新独立泛化测试。
- `predictions/`：原始预测和运行元数据。
- `results/final/`：统一最终评分、字段识别、配对区间与图表。
- `../../logs/stage9/`：运行命令、推理日志、冻结构造器与失败记录。
- `../../scripts/stage9_schema_binding.py`：构造、推理、评分与字段识别数据生成。
- `../../scripts/run_stage9_suite.py`：按单 GPU 顺序执行实验。
- `../../scripts/analyze_stage9_results.py`：任务级与模板级 Bootstrap、免训练对照和诊断统计。

## 复现

在仓库根目录运行。基础模型和既有 LoRA 需已完整恢复；Git LFS 指针不是权重。FCTW 原权重 SHA-256 为 `e408b8230d945288b54bbb07294ebe301bfe4cc3f1c8b666652e7f52b24013d2`，本次从本机备份恢复到单独目录，没有修改历史检查点。

```bash
python -m pytest -q tests/test_stage9_schema_binding.py
python scripts/run_stage9_suite.py direct --fctw-adapter /path/to/restored/fctw
python scripts/run_stage9_suite.py baselines --fctw-adapter /path/to/restored/fctw
python scripts/run_stage9_suite.py field --fctw-adapter /path/to/restored/fctw
```

命令拒绝覆盖已有预测；重跑应在单独的干净副本中执行。推理前检查 GPU 0 至少有 8GB 空闲，进程显存分配上限为 16%。这只能约束本进程显存，不能保证共享算力没有相互影响。

统一评分使用 `stage9_schema_binding.py evaluate`，为每份预测传入 `--prediction 名称=路径`。直接调用名称为 `base`、`fdsrc`、`fctw`；基线名称为 `模型_explicit_mapping` 和 `模型_description_alignment`。最后执行：

```bash
python scripts/analyze_stage9_results.py --root . --result-dir reports/stage9_schema_binding/results/final --output reports/stage9_schema_binding/results/final/evidence.json --plot
```

旧因子数据 v1 及 `base_direct.jsonl`、`base_pilot.jsonl` 属于无效试运行，不参与结论。正式目录为 `stage9_schema_binding_v2`，其内部 `factorial_v1.jsonl` 文件名为兼容入口保留，版本以清单为准。中间评测快照的拒绝与解析分类以 `results/final` 为准。

本次与 Stage 8 的 60 个请求精确无重复；不宣称预训练去污染。6 个工具模板在 pilot 与正式集之间共享，不宣称未见工具家族泛化。
