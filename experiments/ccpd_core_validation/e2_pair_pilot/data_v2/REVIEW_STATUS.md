# v2 审核状态

该版本实际用于 Base/FDSRC 原生推理和 8 对 Rollout 预检，数据哈希为：

`e5b5eedcf104396cc11c358904ba873b09989da48d86825b7191aebaadf7c70d`

分层抽查 14 对后，发现 `ccpd_fctw_multiple_186` 的请求使用 `under 30 minutes`，来源 Gold 为 `time=30`。由于自然语言严格小于与 JSON Schema `maximum=30` 的闭区间语义存在歧义，该样本不能作为高可信训练 Gold。

因此 v2 的语义抽查为 13/14 明确、1/14 有歧义。实验结果保留，不回写；修正规则后的数据另存为 v3。
