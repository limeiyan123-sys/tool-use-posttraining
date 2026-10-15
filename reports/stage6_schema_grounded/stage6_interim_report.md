# Stage 6：Schema-Grounded Capability Learning 阶段实验报告

## 1. 研究问题

本阶段研究动态工具环境中的失败究竟主要来自：

1. 模型无法理解工具能力约束；
2. 模型能够识别冲突，但不能把判断稳定转化为调用或拒绝；
3. 用户需求没有正确绑定到当前 Schema 字段；
4. 参数生成错误掩盖了工具决策能力。

所有结论均来自已有冻结预测或本阶段真实推理结果。本阶段没有重新训练模型，也没有修改 Gold Answer。

## 2. FWCT 错误归因

在 300 条动态工具世界评测中，FDSRC、FWCT、FCTW 的端到端 Decision Exact 分别为 52.67%、62.33%、54.00%。

### Capability Substitution（60 条）

| 方法 | 正确 | False Call | Wrong Tool | False Reject |
|---|---:|---:|---:|---:|
| FDSRC | 0 | 46 | 14 | 0 |
| FWCT | 34 | 12 | 13 | 1 |
| FCTW | 2 | 44 | 14 | 0 |

### Distractor（60 条）

| 方法 | 正确 | False Reject | Wrong Tool | Argument Failure | Invalid |
|---|---:|---:|---:|---:|---:|
| FDSRC | 36 | 0 | 6 | 17 | 1 |
| FWCT | 32 | 14 | 3 | 11 | 0 |
| FCTW | 40 | 0 | 3 | 16 | 1 |

FWCT 的 Distractor 退化主要由错误拒绝增加导致，而不是参数错误恶化。FWCT 学到了更强的拒绝倾向，但没有形成稳定的 Valid/Near-miss 判别边界。FCTW 恢复了 Distractor 调用倾向，却基本丢失 FWCT 在 Capability Substitution 上的收益，说明四象限平衡仍可能只是在调用与拒绝偏置之间折中。

## 3. 单工具能力探针

探针包含 30 个基础意图，每个意图构造 Valid、Near-miss、功能不匹配三类单工具环境，共 90 条。约束覆盖 enum、numeric range、训练未见的 string pattern，以及功能不匹配。

| 模型 | Direct Overall | Explicit Overall | Near-miss Direct | Near-miss Explicit | Valid Direct | Valid Explicit |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 58.89% | 76.67% | 0.00% | 100.00% | 93.33% | 30.00% |
| LoRA | 63.33% | 68.89% | 0.00% | 100.00% | 93.33% | 6.67% |
| FDSRC | 64.44% | 77.78% | 0.00% | 100.00% | 93.33% | 40.00% |
| FWCT | 72.22% | 73.33% | 46.67% | 86.67% | 70.00% | 36.67% |
| FCTW | 66.67% | 77.78% | 13.33% | 96.67% | 86.67% | 40.00% |

Near-miss 在显式判断中达到 86.67% 至 100.00%，但在原调用协议中显著下降，说明模型能够被提示激活冲突信号，却不能稳定把信号整合进最终行动。另一方面，约束精确定位最高仅 26.67%，字段定位最高 45.00%，不能把显式二分类的高准确率解释为完整的 Schema 语义理解。

## 4. 确定性 Schema 校验实验

实现严格的调用后校验器：只接受可解析的正式输出；删除未知工具或参数违反当前 Schema 的调用；不新增调用、不猜测缺失需求、不修复参数，也不宽松解析非法输出。

### 动态工具世界结果

| 方法 | 原始 Decision Exact | 过滤后 | 变化 | Substitution 原始 | 过滤后 | Correct→Wrong |
|---|---:|---:|---:|---:|---:|---:|
| FDSRC | 52.67% | 67.33% | +14.66 pp | 0.00% | 66.67% | 0 |
| FWCT | 62.33% | 68.00% | +5.67 pp | 56.67% | 80.00% | 0 |
| FCTW | 54.00% | 67.67% | +13.67 pp | 3.33% | 66.67% | 0 |

Original 与 Distractor 的端到端准确率均保持不变。三种模型分别拦截 86、39、72 个调用，修复 44、17、41 个原始错误，没有把端到端正确样本改错。这证明当前工具 Schema 中的形式化能力边界可以可靠参与最终决策，而且大量 Substitution 失败无需再次训练即可修复。

### BFCL holdout 回归

| 方法 | 指标 | 原始 | 过滤后 | 变化 |
|---|---|---:|---:|---:|
| FDSRC | Tool Accuracy | 100.00% | 97.14% | -2.86 pp |
| FDSRC | Arg Accuracy | 93.94% | 91.87% | -2.07 pp |
| FDSRC | Arg Exact | 81.43% | 81.43% | 0.00 pp |
| FCTW | Tool Accuracy | 97.14% | 92.86% | -4.29 pp |
| FCTW | Arg Accuracy | 92.87% | 90.08% | -2.79 pp |
| FCTW | Arg Exact | 84.29% | 84.29% | 0.00 pp |

被拦截的 BFCL 调用原本均已属于参数错误，例如数组元素类型错误或缺少必填字段，因此没有端到端正确样本被破坏，Arg Exact 也未下降。但错误从 Argument Failure 转移为 Tool Selection Error，使 Tool Accuracy 与平均参数指标下降。硬过滤适合阻止动态环境中的越界调用，却不是普通工具调用协议下的无损后处理。

## 5. 核心结论

1. 小模型并非完全看不懂 Near-miss；主要断点是冲突证据没有稳定进入调用决策。
2. FWCT 的收益与退化可以由拒绝门控偏置解释；FCTW 的平衡数据尚未形成稳定、可迁移的能力判别规则。
3. 形式化 Schema 约束应交给确定性程序执行。该方法以零训练成本显著改善 Capability Substitution，是当前最简且最有解释力的工程方案。
4. 剩余错误主要是 Requirement-to-Schema Binding：用户明确需求对应可选字段时，模型可能直接省略该字段；校验器无法判断“语法合法但语义不完整”的调用。
5. 当前证据不足以支持继续训练 Capability Certificate，也不足以宣称独立泛化已解决。单工具探针和动态工具世界都来自历史诊断体系，仍需新的冻结测试集验证。

## 6. 研究决策

本阶段暂不追加 LoRA SFT。下一步冻结新的独立工具世界，优先实现轻量的用户需求到 Schema 字段绑定，再用同一确定性校验器验证。若语义绑定仍无法改善剩余的 Wrong Tool 与 Argument Failure，再构造小规模 Decision-Linked SFT；不引入额外分类头、复杂损失或 RL。

当前方法应准确表述为 **Schema-Grounded Capability Integration**，它是模型语义推理与程序化约束验证的混合框架，而不是已验证的新型训练算法。
