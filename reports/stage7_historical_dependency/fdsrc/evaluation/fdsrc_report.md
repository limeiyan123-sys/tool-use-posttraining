# Stage 7 fdsrc 配对评测

- 基础任务数：24
- Original Accuracy：100.00%
- Updated Accuracy：100.00%
- Updated - Original：+0.00 pp
- Pair Success Rate：100.00%
- Stale-Choice Rate（全部 Updated）：0/24 = 0.00%
- Stale-Choice Rate（Original 正确条件下）：0/24 = 0.00%
- Semantic Error Rate：0.00%
- False Reject Rate：0.00%

## 配对 Bootstrap 95% 区间

| 指标 | 下界 | 上界 |
|---|---:|---:|
| original_accuracy | 100.00% | 100.00% |
| updated_accuracy | 100.00% | 100.00% |
| updated_minus_original | 0.00% | 0.00% |
| pair_success_rate | 100.00% | 100.00% |
| stale_choice_rate_all_updated | 0.00% | 0.00% |

## 分工具家族

| 家族 | Original | Updated | Stale Choice |
|---|---:|---:|---:|
| archive_pack | 100.00% | 100.00% | 0.00% |
| chart_render | 100.00% | 100.00% | 0.00% |
| document_translate | 100.00% | 100.00% | 0.00% |
| image_convert | 100.00% | 100.00% | 0.00% |
| invoice_currency | 100.00% | 100.00% | 0.00% |
| report_export | 100.00% | 100.00% | 0.00% |

## 解释边界

- Pilot 只有 24 个配对任务，Bootstrap 区间用于表达不确定性，不宣称统计显著。
- Semantic Error 仅指格式合法、参数可比较但工具能力不满足请求的调用。
- Mock Tool 依据冻结能力契约执行，不使用模型自由判断兼容性。
