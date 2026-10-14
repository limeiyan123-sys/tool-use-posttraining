# SFT 数据说明

本目录的 `manifest.json` 记录数据规模、构造策略、随机种子和 SHA-256。完整 `train.json` 与 `dev.json` 由脚本在实验服务器生成，不提交到 GitHub，原因如下：

1. 避免重复分发上游训练数据。
2. 上游样本中存在形似访问令牌的示例字符串，即使它们只是数据内容，也不适合公开提交。
3. 控制仓库体积，并让数据生成与污染检查成为可审计流程。

使用 `scripts/prepare_toolace_sft.py`、`scripts/build_robust_sft.py`、`scripts/build_counterfactual_sft.py` 和 `scripts/build_failure_weighted_sft.py` 可依次重建各阶段数据。

