# CCPD 核心机制验证

本目录用于验证合同反事实配对决策学习（CCPD）的核心研究假设。各实验必须区分解析目标、采样奖励、GRPO 实际优势以及模型行为，不把计划阈值当作实验结果。

当前状态：E0、E1、E2 与 E3 训练前奖励预检已完成。由于 B3 联合奖励在 8 对实测中出现 100% 零方差组，按预注册止损规则未启动参数训练；阶段判断为 `REVISE`。

## 目录

- `e0_theory/`：配对目标、平衡逐样本目标、GRPO 分组和代价敏感目标的理论与数值检查。
- `e1_diagnosis/`：21 对严格合同诊断、原生工具调用、双标签顺序受控分数和配对 Bootstrap。
- `e2_pair_pilot/`：合同配对构造、Gold 审计、Base/FDSRC 初态错误和 8-Rollout 奖励稀疏性预检。

## 复现

```bash
python experiments/ccpd_core_validation/e0_theory/ccpd_e0_analysis.py
python experiments/ccpd_core_validation/e1_diagnosis/audit_pairs.py \
  --project-root . --output-dir /tmp/ccpd_e1_audit
python experiments/ccpd_core_validation/e2_pair_pilot/build_e2_pilot.py \
  --project-root . --output-dir /tmp/ccpd_e2_pilot --pairs 60
```

E0 与数据构造脚本不加载语言模型。E1/E2 的模型脚本只读加载既有权重，不修改 Stage 0–9 的历史数据与预测。
