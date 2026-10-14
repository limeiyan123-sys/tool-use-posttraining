# diagnostic_comparison

## 分工具世界结果

| 方法 | Overall | Original | Ephemeral | Removal | Substitution | Distractor | Interface Equiv. | All-worlds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 50.67% | 71.33% | 48.67% | 66.67% | 0.00% | 66.67% | 48.67% | 0.00% |
| LoRA SFT | 54.13% | 74.00% | 53.33% | 75.33% | 0.67% | 67.33% | 53.33% | 0.67% |
| Schema-only | 55.87% | 72.67% | 63.33% | 75.33% | 0.00% | 68.00% | 63.33% | 0.00% |
| FDSRC | 54.67% | 74.67% | 61.33% | 74.67% | 0.67% | 62.00% | 61.33% | 0.67% |
| World Randomization | 60.00% | 76.00% | 70.67% | 76.67% | 3.33% | 73.33% | 70.67% | 2.67% |
| Counterfactual Paired | 58.67% | 76.00% | 67.33% | 75.33% | 0.67% | 74.00% | 67.33% | 0.67% |
| Failure-Weighted CTW | 64.13% | 72.67% | 64.67% | 77.33% | 47.33% | 58.67% | 64.67% | 14.00% |

## 成对 Bootstrap

### Failure-Weighted CTW 相对 FDSRC

| 范围 | Decision Exact 差值 | 95% CI | 样本数 |
|---|---:|---:|---:|
| overall | +9.47 pp | [+6.40, +12.53] pp | 750 |
| original | -2.00 pp | [-6.67, +2.00] pp | 150 |
| ephemeral_interface | +3.33 pp | [-2.00, +9.33] pp | 150 |
| capability_removal | +2.67 pp | [-0.67, +6.67] pp | 150 |
| capability_substitution | +46.67 pp | [+38.67, +54.67] pp | 150 |
| distractor | -3.33 pp | [-12.00, +5.33] pp | 150 |

#### 真实案例

- 改善 `tw_multiple_108__capability_substitution`（capability_substitution）：参考错误 `incompatible_substitute_selection`，候选错误 `none`。
  - 用户请求：Calculate the heat capacity at constant pressure for air, given its temperature is 298K and volume is 10 m^3.
  - Gold：`[]`
  - 参考预测：`[{"arguments": {"gas": "air", "temp": 298, "volume": 10}, "name": "limited_708d5fb525"}]`
  - 候选预测：`[]`
- 改善 `tw_multiple_108__distractor`（distractor）：参考错误 `distractor_selection_error`，候选错误 `none`。
  - 用户请求：Calculate the heat capacity at constant pressure for air, given its temperature is 298K and volume is 10 m^3.
  - Gold：`[{"arguments": {"gas": ["air", ""], "temp": [298], "volume": [10]}, "name": "calc_heat_capacity"}]`
  - 参考预测：`[{"arguments": {"gas": "air", "temp": 298, "volume": 10}, "name": "distractor_823f47b848"}]`
  - 候选预测：`[{"arguments": {"gas": "air", "temp": 298, "volume": 10}, "name": "calc_heat_capacity"}]`
- 退化 `tw_multiple_193__ephemeral_interface`（ephemeral_interface）：参考错误 `none`，候选错误 `unnecessary_refusal`。
  - 用户请求：Get me the travel distance and duration from the Eiffel Tower to the Louvre Museum
  - Gold：`[{"arguments": {"arg_37b6ced1e5": ["", false], "arg_6bdb8bf67f": ["Eiffel Tower"], "arg_6fe5f1495e": ["Louvre Museum"]}, "name": "tool_5f42d68c71"}]`
  - 参考预测：`[{"arguments": {"arg_6bdb8bf67f": "Eiffel Tower", "arg_6fe5f1495e": "Louvre Museum"}, "name": "tool_5f42d68c71"}]`
  - 候选预测：`[]`

### Counterfactual Paired 相对 World Randomization

| 范围 | Decision Exact 差值 | 95% CI | 样本数 |
|---|---:|---:|---:|
| overall | -1.33 pp | [-2.67, +0.00] pp | 750 |
| original | +0.00 pp | [-2.67, +2.67] pp | 150 |
| ephemeral_interface | -3.33 pp | [-7.33, +0.67] pp | 150 |
| capability_removal | -1.33 pp | [-3.33, +0.00] pp | 150 |
| capability_substitution | -2.67 pp | [-5.33, -0.67] pp | 150 |
| distractor | +0.67 pp | [-2.67, +4.00] pp | 150 |

#### 真实案例

- 改善 `tw_multiple_120__distractor`（distractor）：参考错误 `distractor_selection_error`，候选错误 `none`。
  - 用户请求：How much time will it take for the light to reach earth from a star 4 light years away?
  - Gold：`[{"arguments": {"distance_in_light_years": [4], "speed_of_light": [299792458, ""]}, "name": "light_travel_time"}]`
  - 参考预测：`[{"arguments": {"distance_in_light_years": 4}, "name": "distractor_88f193c6f6"}]`
  - 候选预测：`[{"arguments": {"distance_in_light_years": 4}, "name": "light_travel_time"}]`
- 改善 `tw_parallel_105__ephemeral_interface`（ephemeral_interface）：参考错误 `interface_argument_mapping_error`，候选错误 `none`。
  - 用户请求："Can you calculate the heat capacity at constant pressure of air for a science experiment I am conducting? I have a container with a volume of 2.5 m^3 and I am able to maintain the temperature at 300 Kelvin. I will be repeating the experiment at a higher temperature of 350 Kelvin and then at a lower volume of 1.5 m^3. I am using air for all these experiments. Can you provide the heat capacity for these three different conditions?"
  - Gold：`[{"arguments": {"arg_a443591174": [300], "arg_baed2f20ea": [2.5], "arg_df48871d83": ["air", ""]}, "name": "tool_85b0ba8d8a"}, {"arguments": {"arg_a443591174": [350], "arg_baed2f20ea": [2.5], "arg_df48871d83": ["air", ""]}, "name": "tool_85b0ba8d8a"}, {"arguments": {"arg_a443591174": [300], "arg_baed2f20ea": [1.5], "arg_df48871d83": ["air", ""]}, "name": "tool_85b0ba8d8a"}]`
  - 参考预测：`[{"arguments": {"arg_a443591174": 300, "arg_baed2f20ea": 2.5, "arg_df48871d83": "air"}, "name": "tool_85b0ba8d8a"}, {"arguments": {"arg_a443591174": 350, "arg_baed2f20ea": 2.5, "arg_df48871d83": "air"}, "name": "tool_85b0ba8d8a"}, {"arguments": {"arg_a443591174": 1.5, "arg_baed2f20ea": 2.5, "arg_df48871d83": "air"}, "name": "tool_85b0ba8d8a"}]`
  - 候选预测：`[{"arguments": {"arg_a443591174": 300, "arg_baed2f20ea": 2.5, "arg_df48871d83": "air"}, "name": "tool_85b0ba8d8a"}, {"arguments": {"arg_a443591174": 350, "arg_baed2f20ea": 2.5, "arg_df48871d83": "air"}, "name": "tool_85b0ba8d8a"}, {"arguments": {"arg_a443591174": 300, "arg_baed2f20ea": 1.5, "arg_df48871d83": "air"}, "name": "tool_85b0ba8d8a"}]`
- 退化 `tw_multiple_167__ephemeral_interface`（ephemeral_interface）：参考错误 `none`，候选错误 `interface_argument_mapping_error`。
  - 用户请求：Generate a melody in C major scale, starting with the note C4, 16 measures long, at 120 beats per minute.
  - Gold：`[{"arguments": {"arg_01f76fe559": [120, ""], "arg_7a4733f7a2": [16], "arg_f1789c8ac0": ["C4"], "arg_fc7beb3437": ["C"]}, "name": "tool_d4ae4461ea"}]`
  - 参考预测：`[{"arguments": {"arg_01f76fe559": 120, "arg_7a4733f7a2": 16, "arg_f1789c8ac0": "C4", "arg_fc7beb3437": "C"}, "name": "tool_d4ae4461ea"}]`
  - 候选预测：`[{"arguments": {"arg_01f76fe559": 120, "arg_7a4733f7a2": 16, "arg_fc7beb3437": "C"}, "name": "tool_d4ae4461ea"}]`

### Failure-Weighted CTW 相对 World Randomization

| 范围 | Decision Exact 差值 | 95% CI | 样本数 |
|---|---:|---:|---:|
| overall | +4.13 pp | [+1.20, +7.07] pp | 750 |
| original | -3.33 pp | [-6.67, +0.00] pp | 150 |
| ephemeral_interface | -6.00 pp | [-11.33, -1.33] pp | 150 |
| capability_removal | +0.67 pp | [-2.67, +4.00] pp | 150 |
| capability_substitution | +44.00 pp | [+36.00, +52.00] pp | 150 |
| distractor | -14.67 pp | [-22.00, -8.00] pp | 150 |

#### 真实案例

- 改善 `tw_multiple_108__capability_substitution`（capability_substitution）：参考错误 `incompatible_substitute_selection`，候选错误 `none`。
  - 用户请求：Calculate the heat capacity at constant pressure for air, given its temperature is 298K and volume is 10 m^3.
  - Gold：`[]`
  - 参考预测：`[{"arguments": {"gas": "air", "temp": 298, "volume": 10}, "name": "limited_708d5fb525"}]`
  - 候选预测：`[]`
- 改善 `tw_multiple_120__distractor`（distractor）：参考错误 `distractor_selection_error`，候选错误 `none`。
  - 用户请求：How much time will it take for the light to reach earth from a star 4 light years away?
  - Gold：`[{"arguments": {"distance_in_light_years": [4], "speed_of_light": [299792458, ""]}, "name": "light_travel_time"}]`
  - 参考预测：`[{"arguments": {"distance_in_light_years": 4}, "name": "distractor_88f193c6f6"}]`
  - 候选预测：`[{"arguments": {"distance_in_light_years": 4}, "name": "light_travel_time"}]`
- 退化 `tw_multiple_167__ephemeral_interface`（ephemeral_interface）：参考错误 `none`，候选错误 `interface_argument_mapping_error`。
  - 用户请求：Generate a melody in C major scale, starting with the note C4, 16 measures long, at 120 beats per minute.
  - Gold：`[{"arguments": {"arg_01f76fe559": [120, ""], "arg_7a4733f7a2": [16], "arg_f1789c8ac0": ["C4"], "arg_fc7beb3437": ["C"]}, "name": "tool_d4ae4461ea"}]`
  - 参考预测：`[{"arguments": {"arg_01f76fe559": 120, "arg_7a4733f7a2": 16, "arg_f1789c8ac0": "C4", "arg_fc7beb3437": "C"}, "name": "tool_d4ae4461ea"}]`
  - 候选预测：`[{"arguments": {"arg_01f76fe559": 120, "arg_7a4733f7a2": 16, "arg_fc7beb3437": "C"}, "name": "tool_d4ae4461ea"}]`

## 统计说明

- Bootstrap 重采样次数：10000。
- 随机种子：20261020。
- 区间是样本级成对 Bootstrap 的经验 95% 区间，不对多重比较作校正。
- 非法输出保持失败，不做自动修复。
