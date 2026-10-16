"""对自由文本类别值的精确匹配歧义做离线敏感性分析，不改 Gold。"""

import argparse
from pathlib import Path

from stage9_schema_binding import aggregate_model, read_jsonl, write_json


def main():
    """公开报告删除整个商品类别模板后的成对结果，避免挑选个别输出同义词。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    for model in ("base", "fdsrc", "fctw"):
        cases = read_jsonl(args.results / f"{model}_cases.jsonl")
        selected = [case for case in cases if case["family"] != "catalog_filter"]
        summaries[model] = aggregate_model(selected, 5000)
    write_json(args.results / "gold_sensitivity.json", {
        "reason": "商品类别为自由字符串，headphones/耳机、mouse/鼠标等同义输出在严格 Gold 下被判错；缺少真实 API 不能断言执行失败。",
        "method": "事后敏感性分析，固定排除完整 catalog_filter 家族，三模型四条件同步保留其余20任务；不重跑模型，不修改原Gold，原24任务指标保留。",
        "summaries": summaries,
    })
    print({model: {condition: round(metrics["decision_exact"] * 100, 2) for condition, metrics in summary["by_condition"].items()} for model, summary in summaries.items()})


if __name__ == "__main__":
    main()
