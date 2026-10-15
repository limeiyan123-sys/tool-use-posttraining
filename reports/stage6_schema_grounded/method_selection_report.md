# Stage 6 Method Selection Report

## 选择

本阶段选择 **Schema-Grounded Capability Integration**，采用“模型语义绑定 + 确定性 Schema 校验”的最小混合方案。

第一步已经实现 Schema Symbolic Verification：对模型提出的每个调用执行确定性 Schema 校验，只删除未知工具或参数违反当前 Schema 的调用，不猜测、不补全、不修复参数。

动态工具世界的实测结果表明，该操作可以直接修复大量 Capability Substitution 错误，同时不改变 Original 与 Distractor 的端到端准确率。因此，当前没有证据支持立即增加一次 SFT；下一步应先解决校验器无法观察到的 Requirement-to-Schema Binding。

## 为什么不先做 Capability Certificate SFT

- 五个模型在显式二分类下已经能识别大多数 Near-miss，新增二分类监督不直接针对主要断点。
- 约束精确定位接近零，直接训练完整证书会同时混入格式学习、字段绑定和行动学习，难以解释收益来源。
- 显式提示对有效工具存在过度拒绝，若直接把该输出作为训练目标，可能重复 FWCT 的拒绝偏置。

## 为什么保留 Symbolic Baseline

- Enum、Range、Pattern 等形式化约束应由确定性程序可靠验证，不应默认要求 1.5B 模型学习规则执行。
- 但只校验模型已生成的参数无法发现“用户明确要求 Patent，模型却省略可选 case_type”这类需求遗漏。
- 因此 Symbolic Baseline 既是强对照，也能明确剩余问题是否属于 Requirement-to-Schema Binding。

## 已验证的作用与边界

- FDSRC、FWCT、FCTW 的动态工具世界 Decision Exact 分别提高 14.66、5.67、13.67 个百分点。
- Capability Substitution 分别从 0.00%、56.67%、3.33% 提高到 66.67%、80.00%、66.67%。
- 三个模型均未出现原本端到端正确、过滤后错误的样本。
- 在 70 条 BFCL holdout 上，Arg Exact 保持不变，但 Tool Accuracy 因错误类型迁移下降 2.86 至 4.29 个百分点；硬过滤不能被宣称为普通工具调用的无损替代。
- 对用户要求映射到可选字段、同名字段语义绑定以及多调用部分完成等问题，单纯 Schema 校验无法处理。

## 最小后续实验

冻结一组新的独立工具世界，只实现“用户需求抽取到 Schema 字段”的轻量语义绑定层，再由相同的确定性校验器验证。只有该方法仍不能解决剩余错误时，才构造少量 Decision-Linked SFT；不增加证书长度、复杂损失或额外分类头。
