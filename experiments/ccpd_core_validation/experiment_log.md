# CCPD 核心验证实验台账

## E0 理论与梯度检查

```yaml
experiment_name: ccpd_e0_theory_gradient_check
model: logistic_toy_model_cpu
dataset: analytical_scenarios_and_exact_binary_enumeration
method: balanced_independent_vs_paired_product_vs_grpo_grouping
hyperparameter:
  pairs_per_group: 4
  probability_scenarios: 6
  cost_alpha: [0.3, 0.5, 0.7]
result:
  autograd_max_abs_error: 6.938893903907228e-17
  shared_bias_stationary: p_plus_plus_p_minus_equals_1
  reverse_separation_pair_nonzero_group_probability: 0.0394
analysis: 配对目标独特性质是跨侧门控，不是独有的等错误率驻点；双错初态存在严重奖励稀疏。
next_step: 继续 E1 零训练诊断，不直接训练 v1 M2。
```

## E1 合同判别与调用倾向

```yaml
experiment_name: ccpd_e1_contract_sensitivity_diagnosis
model:
  - Qwen2.5-1.5B-Instruct
  - FDSRC_LoRA
  - FWCT_LoRA
  - FCTW_LoRA
dataset: capability_probe_v1_valid_near_miss_pairs
method: native_tool_call_plus_dual_order_single_token_probe
hyperparameter:
  strict_pairs: 21
  sensitivity_pairs: 30
  decoding: greedy_for_native
  bootstrap_repeats: 10000
  seed: 20261017
result:
  strict_native_pair_success:
    base: 0.0
    fdsrc: 0.0
    fwct: 0.095238
    fctw: 0.047619
  strict_probe_concordance:
    base: 1.0
    fdsrc: 1.0
    fwct: 1.0
    fctw: 1.0
  fwct_minus_fdsrc_score_difference: 0.244048
  fwct_minus_fdsrc_mean_score: -0.345238
analysis: FWCT 同时增强少量合同分数差并降低整体调用倾向；所有模型已有强排序，但原生动作未稳定使用该信号。
next_step: 构造更严格且无编辑痕迹捷径的 E2 配对，检查真实奖励信号。
```

## E2 合同配对 Pilot

```yaml
experiment_name: ccpd_e2_pair_gold_pilot
model: no_training
dataset: fctw_v1_source_tasks_rebuilt_as_minimal_pairs
method: required_explicit_target_plus_symmetric_constraint_mutation
hyperparameter:
  source_tasks: 100
  frozen_pairs: 60
  split: [40, 10, 10]
  seed: 20261017
result:
  accepted_before_v3_sampling: 64
  discarded_optional_target: 16
  discarded_no_direct_evidence: 18
  discarded_strict_boundary_ambiguity: 2
  v3_sha256: e5788137bae54d0c13089023e4a0e36eaa971f9ac03e03ec1f97b3f7b82e549f
  stratified_semantic_spot_check: 14/14
analysis: 合同对可低成本构造，但 v2 的严格边界失败证明 Schema 自洽不能代替语义审核；14 对抽查不足以证明总体 95% 一致率。
next_step: 使用已完成的 v2 初态预测做奖励预检；正式训练仅能使用修订后的 v3 或新独立数据。
```

## E3 训练前 Rollout 预检

```yaml
experiment_name: ccpd_e3_reward_signal_preflight
model: Qwen2.5-1.5B-Instruct
dataset: ccpd_e2_pilot_v2_train_first_8_pairs
method: sampled_native_tool_call_and_exact_b1_b2_b3_reward_analysis
hyperparameter:
  pairs: 8
  rollouts_per_side: 4
  temperature: 0.8
  top_p: 0.95
  max_new_tokens: 96
  seed: 20261017
result:
  plus_decision_success: 0.84375
  plus_content_success: 0.375
  minus_decline_success: 0.0
  joint_pair_success: 0.0
  b2_zero_variance_groups: 0.5
  b3_zero_variance_groups: 1.0
  b2_effective_advantage_fraction: 0.5
  b3_effective_advantage_fraction: 0.0
analysis: 当前 M2 联合奖励在真实初态完全没有 GRPO 学习信号；相同 Rollout 的 B2 仍有有效优势。
next_step: 按止损规则不启动 B0-B3 参数训练；研究决策为 REVISE。
```

## 失败与修复记录

1. E2 v1 的 `restriction_marker_distinguishes` 使用了 Python 链式比较，错误报告 60/60 可区分。配对内容未受影响；结果保留在 `data/` 并标记弃用，修复版为 v2。
2. v2 分层抽查发现 `under 30` 与 `maximum=30` 的语义边界歧义。加入严格上界措辞舍弃规则后生成 v3；v2 预测和预检按原哈希保留，不回写。
3. PEFT 加载历史 Adapter 时警告四个新版本配置键被旧运行环境忽略。权重结构正常加载，警告保留在实验解释边界中。
4. FDSRC 原生生成较 Base 慢，原因是更多响应生成至 Token 上限；实验未缩短或修复这些输出。
