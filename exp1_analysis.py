import argparse
import json
import os
from statistics import mean, median

from src.utils.plots import plot_metric_boxplot, plot_metric_scatter
from src.utils.records import NUMERIC_FIELDS, load_records_from_csv


def _collect_metric_values(records, metric_key, group_key, group_value):
    values = []
    for record in records:
        if record.get(group_key) != group_value:
            continue
        values.append(float(record[metric_key]))
    return values


def summarize_records(records, group_key="status"):
    """
    生成按指定维度分组的指标摘要。
    """
    if not records:
        raise ValueError("records 不能为空。")
    if group_key not in records[0]:
        raise KeyError(f"记录中不存在分组字段: {group_key}")

    summary = {
        "sample_count": len(records),
        "group_key": group_key,
        "group_counts": {},
        "metrics": {},
    }
    group_values = sorted({record[group_key] for record in records})
    for group_value in group_values:
        summary["group_counts"][group_value] = sum(
            1 for record in records if record[group_key] == group_value
        )

    metric_keys = [
        "cross_bbox_attention_ratio",
        "cross_bbox_mean_attention",
        "vision_bbox_attention_ratio",
        "vision_bbox_mean_attention",
    ]
    for metric_key in metric_keys:
        summary["metrics"][metric_key] = {}
        for group_value in group_values:
            values = _collect_metric_values(records, metric_key, group_key, group_value)
            if not values:
                raise ValueError(
                    f"指标缺失: metric_key={metric_key}, "
                    f"group_key={group_key}, group_value={group_value}"
                )

            summary["metrics"][metric_key][group_value] = {
                "count": len(values),
                "mean": mean(values),
                "median": median(values),
            }

    return summary


def generate_analysis_artifacts(records, output_dir, group_key="status"):
    """
    基于结构化记录生成摘要与图表。

    group_key 用于指定按哪个维度汇总，例如:
    - status: 比较攻击成功与失败
    - patch: 比较加 patch 与不加 patch
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize_records(records, group_key=group_key)

    summary_path = os.path.join(output_dir, f"attention_summary_by_{group_key}.json")
    with open(summary_path, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=4)

    plot_metric_boxplot(
        records,
        "cross_bbox_attention_ratio",
        os.path.join(output_dir, f"cross_attention_ratio_boxplot_by_{group_key}.png"),
        title=f"Cross Attention Ratio Grouped by {group_key}",
        ylabel="Attention Ratio",
        group_key=group_key,
    )
    plot_metric_boxplot(
        records,
        "vision_bbox_attention_ratio",
        os.path.join(output_dir, f"vision_attention_ratio_boxplot_by_{group_key}.png"),
        title=f"Vision Attention Ratio Grouped by {group_key}",
        ylabel="Attention Ratio",
        group_key=group_key,
    )
    plot_metric_boxplot(
        records,
        "cross_bbox_mean_attention",
        os.path.join(output_dir, f"cross_mean_attention_boxplot_by_{group_key}.png"),
        title=f"Cross Mean Attention Grouped by {group_key}",
        ylabel="Attention Strength",
        group_key=group_key,
    )
    plot_metric_boxplot(
        records,
        "vision_bbox_mean_attention",
        os.path.join(output_dir, f"vision_mean_attention_boxplot_by_{group_key}.png"),
        title=f"Vision Mean Attention Grouped by {group_key}",
        ylabel="Attention Strength",
        group_key=group_key,
    )

    return {
        "summary_path": summary_path,
        "output_dir": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze structured attention metrics from experiment 1"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="实验一结果目录，默认使用 data/results/exp1",
    )
    parser.add_argument(
        "--group_key",
        type=str,
        default="status",
        help="分析分组维度，例如 status 或 patch",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(base_dir, "data", "results", "exp1")
    metrics_csv_path = os.path.join(results_dir, "attention_metrics.csv")
    analysis_dir = os.path.join(results_dir, f"analysis_by_{args.group_key}")

    records = load_records_from_csv(metrics_csv_path)
    artifacts = generate_analysis_artifacts(
        records,
        analysis_dir,
        group_key=args.group_key,
    )

    print("=" * 50)
    print(" 实验一分析完成 ")
    print("=" * 50)
    print(f" 分组维度: {args.group_key}")
    print(f" 指标文件: {metrics_csv_path}")
    print(f" 摘要文件: {artifacts['summary_path']}")
    print(f" 图表目录: {artifacts['output_dir']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
