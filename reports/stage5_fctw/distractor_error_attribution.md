# FWCT Distractor 错误归因报告

## 结论

现有证据支持过度拒绝假设。FWCT 减少了参数错误和工具误选，但新增的错误拒绝超过这些收益，使 Distractor 的 Decision Exact 下降。

## 统一协议

- 样本数：60 条独立 Distractor 工具世界。
- Decision Exact：输出可解析、工具集合精确匹配、参数精确匹配且通过 Schema 校验。
- False Rejection：Gold 要求调用工具，但模型未输出任何工具调用。
- Wrong Tool Selection：模型调用了错误工具，包括受限干扰工具。
- Argument Failure：工具选择正确，但参数不完整、错误或不满足 Schema。
- Invalid Output：输出无法按严格 tool_call 协议解析。

## 错误分布

| 方法 | Correct | False Rejection | Wrong Tool | Argument Failure | Invalid Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| FDSRC | 36 | 0 | 6 | 17 | 1 |
| FWCT | 32 | 14 | 3 | 11 | 0 |

- FDSRC Decision Exact：60.00%。
- FWCT Decision Exact：53.33%。
- FWCT 相对 FDSRC 的配对差值：-6.67 pp，95% Bootstrap CI [-20.00, +6.67] pp。
- 错误拒绝率变化：+23.33 pp，95% Bootstrap CI [+13.33, +33.33] pp。

## 配对转移

| FDSRC → FWCT | 数量 |
| --- | ---: |
| `argument_failure -> argument_failure` | 10 |
| `argument_failure -> correct` | 3 |
| `argument_failure -> false_rejection` | 4 |
| `correct -> correct` | 26 |
| `correct -> false_rejection` | 10 |
| `invalid_output -> argument_failure` | 1 |
| `wrong_tool_selection -> correct` | 3 |
| `wrong_tool_selection -> wrong_tool_selection` | 3 |

## 研究判断

1. FWCT 的错误拒绝确实增加：从 0/60 增至 14/60，不是少数孤立案例。
2. FWCT 同时减少参数错误 6 次、工具误选 3 次，说明反事实监督学到了规避不兼容工具的有效信号。
3. 总体退化的直接来源是新增错误拒绝；它把“识别不兼容工具”错误外推为“相似工具出现时倾向拒绝”。
4. Valid × Near-miss 四象限监督能够直接解释并针对这一瓶颈，尤其是 E11 必须 Call、E01 必须 Reject 的对照。
5. 参数错误仍有 11 次，FCTW 不能被预期解决全部参数绑定问题；这部分应作为独立剩余风险保留。

## 代表性真实案例

### 案例 1：正确调用退化为错误拒绝

- ID：`tw_parallel_152__distractor`
- 请求："Can you calculate the result of the following mathematical operation: first, raise the number 3 to the power of 5, then raise the number 2 to the power of 3."
- 正确调用：`[{"arguments": {"base": [2], "exponent": [3], "mod": ["", null]}, "name": "math.power"}, {"arguments": {"base": [3], "exponent": [5], "mod": ["", null]}, "name": "math.power"}]`
- 受限干扰工具：`distractor_f6d387052d`，参数 `base` 仅允许 `7922`。
- FDSRC：`none`；<tool_call>
{"name": "math.power", "arguments": {"base": 3, "exponent": 5}}
</tool_call>
<tool_call>
{"name": "math.power", "arguments": {"base": 2, "exponent": 3}}
</tool_call><|im_end|>
- FWCT：`unnecessary_refusal`；The current tool environment cannot complete this request.<|im_end|>

### 案例 2：参数错误转为错误拒绝

- ID：`tw_simple_python_128__distractor`
- 请求：What's the quarterly dividend per share of a company with 100 million outstanding shares and total dividend payout of 50 million USD?
- 正确调用：`[{"arguments": {"outstanding_shares": [100000000], "total_payout": [50000000]}, "name": "finance.calculate_quarterly_dividend_per_share"}]`
- 受限干扰工具：`distractor_badf458b39`，参数 `outstanding_shares` 仅允许 `100007919`。
- FDSRC：`argument_error`；<tool_call>
{"name": "finance.calculate_quarterly_dividend_per_share", "arguments": {"outstanding_shares": 100007919, "total_payout": 50000000}}
</tool_call><|im_end|>
- FWCT：`unnecessary_refusal`；The current tool environment cannot complete this request.<|im_end|>

### 案例 3：参数错误得到修复

- ID：`tw_multiple_139__distractor`
- 请求：Find details of patent lawsuits involving the company 'Apple Inc.' from the year 2010.
- 正确调用：`[{"arguments": {"case_type": ["Patent"], "company_name": ["Apple Inc."], "year": [2010]}, "name": "lawsuit_details.find"}]`
- 受限干扰工具：`distractor_e0c7f7d8b9`，参数 `case_type` 仅允许 `unsupported_0c161f94`。
- FDSRC：`argument_error`；<tool_call>
{"name": "lawsuit_details.find", "arguments": {"company_name": "Apple Inc.", "year": 2010}}
</tool_call><|im_end|>
- FWCT：`none`；<tool_call>
{"name": "lawsuit_details.find", "arguments": {"case_type": "Patent", "company_name": "Apple Inc.", "year": 2010}}
</tool_call><|im_end|>

### 案例 4：工具误选得到修复

- ID：`tw_parallel_141__distractor`
- 请求：Can you provide me with the list of movie releases in the IMAX format at theaters in New York over the next 7 days, and also the list of movie releases in the 2D format at theaters in Los Angeles over the next 14 days?
- 正确调用：`[{"arguments": {"format": ["IMAX", ""], "location": ["New York", "New York, NY", "NYC"], "timeframe": [7]}, "name": "get_theater_movie_releases"}, {"arguments": {"format": ["2D"], "location": ["Los Angeles", "Los Angeles, CA", "LA"], "timeframe": [14]}, "name": "get_theater_movie_releases"}]`
- 受限干扰工具：`distractor_8a5ff68816`，参数 `location` 仅允许 `unsupported_bb83e44f`。
- FDSRC：`distractor_selection_error`；<tool_call>
{"name": "distractor_8a5ff68816", "arguments": {"format": "IMAX", "location": "New York", "timeframe": 7}}
</tool_call>
<tool_call>
{"name": "distractor_8a5ff68816", "arguments": {"format": "2D", "location": "Los Angeles", "timeframe": 14}}
</tool_call><|im_end|>
- FWCT：`none`；<tool_call>
{"name": "get_theater_movie_releases", "arguments": {"format": "IMAX", "location": "New York", "timeframe": 7}}
</tool_call>
<tool_call>
{"name": "get_theater_movie_releases", "arguments": {"format": "2D", "location": "Los Angeles", "timeframe": 14}}
</tool_call><|im_end|>

## 下一步

诊断支持继续 FCTW-SFT。最简实现应围绕同一意图构造 E10/E01/E11/E00，优先验证 E11 能否降低错误拒绝，同时保持 E01 与 Capability Substitution 的拒绝能力。
不需要在这一阶段增加新的损失函数或强化学习。
