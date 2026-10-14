# FCTW 核心方法比较

| 方法 | Overall | E10 | E01 | E11 | E00 | E11 False Rejection | Unseen Constraint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FDSRC | 60.83% | 80.00% | 0.00% | 63.33% | 100.00% | 0.00% | 60.00% |
| FWCT | 65.00% | 83.33% | 13.33% | 63.33% | 100.00% | 3.33% | 62.50% |
| FCTW | 65.83% | 86.67% | 6.67% | 70.00% | 100.00% | 0.00% | 67.50% |

## 按意图聚类的配对差值

### FCTW - FDSRC

| 指标 | 差值 | 95% CI | 意图组 |
| --- | ---: | ---: | ---: |
| Overall Decision Exact | +5.00 pp | [-0.83, +10.83] pp | 30 |
| E01 Decision Exact | +6.67 pp | [+0.00, +16.67] pp | 30 |
| E11 Decision Exact | +6.67 pp | [-10.00, +23.33] pp | 30 |
| E11 False Rejection | +0.00 pp | [+0.00, +0.00] pp | 30 |
| Unseen Constraint Decision Exact | +7.50 pp | [+0.00, +15.00] pp | 10 |

### FCTW - FWCT

| 指标 | 差值 | 95% CI | 意图组 |
| --- | ---: | ---: | ---: |
| Overall Decision Exact | +0.83 pp | [-3.33, +5.83] pp | 30 |
| E01 Decision Exact | -6.67 pp | [-16.67, +0.00] pp | 30 |
| E11 Decision Exact | +6.67 pp | [+0.00, +16.67] pp | 30 |
| E11 False Rejection | -3.33 pp | [-10.00, +0.00] pp | 30 |
| Unseen Constraint Decision Exact | +5.00 pp | [+0.00, +15.00] pp | 10 |
