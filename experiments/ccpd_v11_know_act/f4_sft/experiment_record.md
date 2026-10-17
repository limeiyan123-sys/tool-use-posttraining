# F4 平衡配对 LoRA SFT 实验记录

experiment_name: `ccpd_v11_f4_balanced_sft`

model: `Qwen2.5-1.5B-Instruct`

dataset: F0 `frozen_v2`，训练 160 对/320 条，开发 4 对/8 条，Call/Reject 比例 1:1；数据 SHA-256 为 `1ff1e8ab5746eecc42510aeac083c12c4c0367e710e419e11f6404cee40b60bf`。

method: 从 Base 初始化的普通 LoRA SFT；可满足侧监督完整函数调用，不可满足侧监督固定拒绝。没有使用新 Loss、RL、Oracle 标签接入或额外分类头。

hyperparameter: rank 8，alpha 16，dropout 0.05，学习率 `1e-4`，单卡 batch 1，梯度累积 8，4 epoch，160 个优化步，BF16，梯度检查点，种子 `20261017`。

git_commit_before_run: `43d69c7a297781aa61d757a4b7c7fbf6b1c315de`

resource: 共享 GPU 1。由于用户明确要求立即运行，跳过原空闲门控；未终止或暂停其他用户进程。训练耗时 289.24 秒。训练期间 `nvidia-smi` 的整卡占用从约 31.4GB 增至约 37.3GB，但该数值混合了其他用户进程，不能作为本进程精确峰值；模型推理阶段框架峰值为 2.986GiB。

result:

- 最佳 checkpoint：step 120；根目录 Adapter 与其 SHA-256 一致。
- 训练损失：0.08291；最佳开发损失：0.02637。
- 训练集 Pair Success 15.00%，False Reject 85.00%。
- Test-ID Pair Success 10.00%，False Reject 90.00%。
- Test-OOD-Family Pair Success 24.00%，False Reject 76.00%。
- Test-OOD-Constraint Pair Success 0.00%，False Reject 100.00%。
- 所有测试不可满足侧 False Call 为 0.00%。
- 训练后后层 Probe：Test-ID AUROC 0.930，OOD-Family 0.923，OOD-Constraint 1.000；三个划分 Pair Concordance 均为 100%。

analysis: 普通平衡 SFT 没有学成稳定的合同条件化策略，而是把 Base 的“总调用”偏置翻转为“多数拒绝”。训练集本身同样失败，不能解释为仅 OOD 泛化不足。Probe 可读性没有随行为塌缩下降，说明失败不能简单归因于合同信息从隐藏表征中消失。Token 级验证损失很低，但序列级首动作错误严重，因此 `eval_loss` 不适合作为该任务唯一的 checkpoint 选择标准。

next_step: 研究首动作概率校准与序列级 Pair Success 的优化不一致，并优先复现或实现明确标注的 Probe&Prefill 强基线。没有新证据前不恢复 CCPD-v1 配对奖励，不直接增加分类头、决策 Token 或复杂 RL。

status: `completed_negative_result`
