# final_comparison

## 分工具世界结果

| 方法 | Overall | Original | Ephemeral | Removal | Substitution | Distractor | Interface Equiv. | All-worlds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FDSRC | 52.67% | 68.33% | 55.00% | 80.00% | 0.00% | 60.00% | 55.00% | 0.00% |
| World Randomization | 55.33% | 71.67% | 56.67% | 80.00% | 3.33% | 65.00% | 56.67% | 1.67% |
| Counterfactual Paired | 54.00% | 68.33% | 53.33% | 80.00% | 1.67% | 66.67% | 53.33% | 0.00% |
| Failure-Weighted CTW | 62.33% | 70.00% | 51.67% | 80.00% | 56.67% | 53.33% | 53.33% | 18.33% |

## 成对 Bootstrap

### Failure-Weighted CTW 相对 FDSRC

| 范围 | Decision Exact 差值 | 95% CI | 样本数 |
|---|---:|---:|---:|
| overall | +9.67 pp | [+4.67, +14.67] pp | 300 |
| original | +1.67 pp | [-5.00, +8.33] pp | 60 |
| ephemeral_interface | -3.33 pp | [-11.67, +5.00] pp | 60 |
| capability_removal | +0.00 pp | [+0.00, +0.00] pp | 60 |
| capability_substitution | +56.67 pp | [+43.33, +68.33] pp | 60 |
| distractor | -6.67 pp | [-20.00, +6.67] pp | 60 |

#### 真实案例

- 改善 `tw_multiple_139__distractor`（distractor）：参考错误 `argument_error`，候选错误 `none`。
  - 用户请求：Find details of patent lawsuits involving the company 'Apple Inc.' from the year 2010.
  - Gold：`[{"arguments": {"case_type": ["Patent"], "company_name": ["Apple Inc."], "year": [2010]}, "name": "lawsuit_details.find"}]`
  - 参考预测：`[{"arguments": {"company_name": "Apple Inc.", "year": 2010}, "name": "lawsuit_details.find"}]`
  - 候选预测：`[{"arguments": {"case_type": "Patent", "company_name": "Apple Inc.", "year": 2010}, "name": "lawsuit_details.find"}]`
- 改善 `tw_multiple_139__original`（original）：参考错误 `argument_error`，候选错误 `none`。
  - 用户请求：Find details of patent lawsuits involving the company 'Apple Inc.' from the year 2010.
  - Gold：`[{"arguments": {"case_type": ["Patent"], "company_name": ["Apple Inc."], "year": [2010]}, "name": "lawsuit_details.find"}]`
  - 参考预测：`[{"arguments": {"company_name": "Apple Inc.", "year": 2010}, "name": "lawsuit_details.find"}]`
  - 候选预测：`[{"arguments": {"case_type": "Patent", "company_name": "Apple Inc.", "year": 2010}, "name": "lawsuit_details.find"}]`
- 退化 `tw_parallel_152__distractor`（distractor）：参考错误 `none`，候选错误 `unnecessary_refusal`。
  - 用户请求："Can you calculate the result of the following mathematical operation: first, raise the number 3 to the power of 5, then raise the number 2 to the power of 3."
  - Gold：`[{"arguments": {"base": [2], "exponent": [3], "mod": ["", null]}, "name": "math.power"}, {"arguments": {"base": [3], "exponent": [5], "mod": ["", null]}, "name": "math.power"}]`
  - 参考预测：`[{"arguments": {"base": 3, "exponent": 5}, "name": "math.power"}, {"arguments": {"base": 2, "exponent": 3}, "name": "math.power"}]`
  - 候选预测：`[]`

### Counterfactual Paired 相对 World Randomization

| 范围 | Decision Exact 差值 | 95% CI | 样本数 |
|---|---:|---:|---:|
| overall | -1.33 pp | [-3.67, +1.00] pp | 300 |
| original | -3.33 pp | [-13.33, +5.00] pp | 60 |
| ephemeral_interface | -3.33 pp | [-10.00, +3.33] pp | 60 |
| capability_removal | +0.00 pp | [+0.00, +0.00] pp | 60 |
| capability_substitution | -1.67 pp | [-5.00, +0.00] pp | 60 |
| distractor | +1.67 pp | [+0.00, +5.00] pp | 60 |

#### 真实案例

- 改善 `tw_parallel_125__original`（original）：参考错误 `argument_error`，候选错误 `none`。
  - 用户请求：In the African savannah, a group of researchers have been observing a herd of elephants for a few years. They have noticed that the current population of elephants is 500 and the annual population growth rate is 2%. They are interested in knowing the estimated population of elephants in 10 years. However, due to the unpredictable nature of the wild, they also want to consider a scenario where the growth rate drops to 1.5% and another scenario where it increases to 2.5%. Can you provide the estimated elephant population for these three scenarios in 10 years?
  - Gold：`[{"arguments": {"current_population": [500], "growth_rate": [0.02], "years": [10]}, "name": "elephant_population_estimate"}, {"arguments": {"current_population": [500], "growth_rate": [0.015], "years": [10]}, "name": "elephant_population_estimate"}, {"arguments": {"current_population": [500], "growth_rate": [0.025], "years": [10]}, "name": "elephant_population_estimate"}]`
  - 参考预测：`[{"arguments": {"current_population": 500, "growth_rate": 2.0, "years": 10}, "name": "elephant_population_estimate"}, {"arguments": {"current_population": 500, "growth_rate": 1.5, "years": 10}, "name": "elephant_population_estimate"}, {"arguments": {"current_population": 500, "growth_rate": 2.5, "years": 10}, "name": "elephant_population_estimate"}]`
  - 候选预测：`[{"arguments": {"current_population": 500, "growth_rate": 0.02, "years": 10}, "name": "elephant_population_estimate"}, {"arguments": {"current_population": 500, "growth_rate": 0.015, "years": 10}, "name": "elephant_population_estimate"}, {"arguments": {"current_population": 500, "growth_rate": 0.025, "years": 10}, "name": "elephant_population_estimate"}]`
- 改善 `tw_parallel_51__distractor`（distractor）：参考错误 `argument_error`，候选错误 `none`。
  - 用户请求：Search for a Chicken Noodle Soup recipe and a Vegan Salad recipe.
  - Gold：`[{"arguments": {"diet": ["", "Keto"], "dish": ["Chicken Noodle Soup"]}, "name": "recipe_search.find"}, {"arguments": {"diet": ["Vegan"], "dish": ["Salad", "salad", "Vegan Salad", "vegan salad"]}, "name": "recipe_search.find"}]`
  - 参考预测：`[{"arguments": {"dish": "Chicken Noodle Soup"}, "name": "recipe_search.find"}, {"arguments": {"dish": "Vegan Salad"}, "name": "recipe_search.find"}]`
  - 候选预测：`[{"arguments": {"diet": "Keto", "dish": "Chicken Noodle Soup"}, "name": "recipe_search.find"}, {"arguments": {"diet": "Vegan", "dish": "Salad"}, "name": "recipe_search.find"}]`
- 退化 `tw_multiple_189__original`（original）：参考错误 `none`，候选错误 `argument_error`。
  - 用户请求：Convert time 3pm from New York time zone to London time zone.
  - Gold：`[{"arguments": {"from_timezone": ["America/New_York", "New York", "New York, NY", "NY", "NYC", "Eastern Standard Time", "EST"], "time": ["3pm"], "to_timezone": ["Europe/London", "London", "British Summer Time", "BST", "Greenwich Mean Time", "GMT"]}, "name": "timezone.convert"}]`
  - 参考预测：`[{"arguments": {"from_timezone": "New York", "time": "3pm", "to_timezone": "London"}, "name": "timezone.convert"}]`
  - 候选预测：`[{"arguments": {"from_timezone": "America/New_York", "time": "15:00", "to_timezone": "Europe/London"}, "name": "timezone.convert"}]`

### Failure-Weighted CTW 相对 World Randomization

| 范围 | Decision Exact 差值 | 95% CI | 样本数 |
|---|---:|---:|---:|
| overall | +7.00 pp | [+2.33, +11.33] pp | 300 |
| original | -1.67 pp | [-8.33, +5.00] pp | 60 |
| ephemeral_interface | -5.00 pp | [-13.33, +1.67] pp | 60 |
| capability_removal | +0.00 pp | [+0.00, +0.00] pp | 60 |
| capability_substitution | +53.33 pp | [+41.67, +65.00] pp | 60 |
| distractor | -11.67 pp | [-21.67, -3.33] pp | 60 |

#### 真实案例

- 改善 `tw_multiple_139__original`（original）：参考错误 `argument_error`，候选错误 `none`。
  - 用户请求：Find details of patent lawsuits involving the company 'Apple Inc.' from the year 2010.
  - Gold：`[{"arguments": {"case_type": ["Patent"], "company_name": ["Apple Inc."], "year": [2010]}, "name": "lawsuit_details.find"}]`
  - 参考预测：`[{"arguments": {"company_name": "Apple Inc.", "year": 2010}, "name": "lawsuit_details.find"}]`
  - 候选预测：`[{"arguments": {"case_type": "Patent", "company_name": "Apple Inc.", "year": 2010}, "name": "lawsuit_details.find"}]`
- 改善 `tw_multiple_189__capability_substitution`（capability_substitution）：参考错误 `incompatible_substitute_selection`，候选错误 `none`。
  - 用户请求：Convert time 3pm from New York time zone to London time zone.
  - Gold：`[]`
  - 参考预测：`[{"arguments": {"from_timezone": "New York", "time": "3pm", "to_timezone": "London"}, "name": "limited_c81630e635"}]`
  - 候选预测：`[]`
- 退化 `tw_parallel_152__distractor`（distractor）：参考错误 `none`，候选错误 `unnecessary_refusal`。
  - 用户请求："Can you calculate the result of the following mathematical operation: first, raise the number 3 to the power of 5, then raise the number 2 to the power of 3."
  - Gold：`[{"arguments": {"base": [2], "exponent": [3], "mod": ["", null]}, "name": "math.power"}, {"arguments": {"base": [3], "exponent": [5], "mod": ["", null]}, "name": "math.power"}]`
  - 参考预测：`[{"arguments": {"base": 3, "exponent": 5}, "name": "math.power"}, {"arguments": {"base": 2, "exponent": 3}, "name": "math.power"}]`
  - 候选预测：`[]`

## 统计说明

- Bootstrap 重采样次数：10000。
- 随机种子：20261020。
- 区间是样本级成对 Bootstrap 的经验 95% 区间，不对多重比较作校正。
- 非法输出保持失败，不做自动修复。
