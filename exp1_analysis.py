import argparse
import json
import os
from statistics import mean, median

from src.utils.plots import plot_metric_boxplot, plot_metric_scatter
from src.utils.records import load_records_from_csv


# NUMERIC_FIELDS 中各字段的含义:
# - image_width / image_height: 攻击图像尺寸。
# - bbox_left / bbox_top / bbox_right / bbox_bottom: 注入文字区域的像素级边界框。
# - cross_bbox_attention_sum: 交叉注意力中，分配到文字区域的总注意力质量。
# - cross_bbox_attention_ratio: 交叉注意力中，文字区域注意力占整图总注意力的比例。
# - cross_bbox_mean_attention: 交叉注意力中，文字区域内部的平均注意力强度。
# - cross_bbox_patch_coverage: 文字区域在交叉注意力 patch 网格上的覆盖面积。
# - vision_bbox_attention_sum: 视觉自注意力中，分配到文字区域的总注意力质量。
# - vision_bbox_attention_ratio: 视觉自注意力中，文字区域注意力占整图总注意力的比例。
# - vision_bbox_mean_attention: 视觉自注意力中，文字区域内部的平均注意力强度。
# - vision_bbox_patch_coverage: 文字区域在视觉注意力 patch 网格上的覆盖面积。
NUMERIC_FIELDS = {
    "image_width",
    "image_height",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
    "cross_bbox_attention_sum",
    "cross_bbox_attention_ratio",
    "cross_bbox_mean_attention",
    "cross_bbox_patch_coverage",
    "vision_bbox_attention_sum",
    "vision_bbox_attention_ratio",
    "vision_bbox_mean_attention",
    "vision_bbox_patch_coverage",
}

def _collect_metric_values(records, metric_key, status):
    values = []
    for record in records:
        if record.get("status") != status:
            continue
        values.append(float(record[metric_key]))
    return values


def summarize_records(records):
    """
    生成按成功/失败分组的指标摘要。
    """
    summary = {
        "sample_count": len(records),
        "status_counts": {},
        "metrics": {},
    }
    statuses = sorted({record.get("status", "UNKNOWN") for record in records})
    for status in statuses:
        summary["status_counts"][status] = sum(
            1 for record in records if record.get("status") == status
        )

    metric_keys = [
        "cross_bbox_attention_ratio",
        "cross_bbox_mean_attention",
        "vision_bbox_attention_ratio",
        "vision_bbox_mean_attention",
    ]
    for metric_key in metric_keys:
        summary["metrics"][metric_key] = {}
        for status in statuses:
            values = _collect_metric_values(records, metric_key, status)
            if not values:
                raise ValueError(f"指标缺失: metric_key={metric_key}, status={status}")

            summary["metrics"][metric_key][status] = {
                "count": len(values),
                "mean": mean(values),
                "median": median(values),
            }

    return summary


def generate_analysis_artifacts(records, output_dir):
    """
    基于结构化记录生成摘要与图表。
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize_records(records)

    summary_path = os.path.join(output_dir, "attention_summary.json")
    with open(summary_path, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=4)

    plot_metric_boxplot(
        records,
        "cross_bbox_attention_ratio",
        os.path.join(output_dir, "cross_attention_ratio_boxplot.png"),
        title="Cross Attention Ratio in Text Region",
        ylabel="Attention Ratio",
    )
    plot_metric_boxplot(
        records,
        "vision_bbox_attention_ratio",
        os.path.join(output_dir, "vision_attention_ratio_boxplot.png"),
        title="Vision Attention Ratio in Text Region",
        ylabel="Attention Ratio",
    )
    plot_metric_scatter(
        records,
        "cross_bbox_attention_ratio",
        "vision_bbox_attention_ratio",
        os.path.join(output_dir, "cross_vs_vision_ratio_scatter.png"),
        title="Cross vs Vision Attention Ratio",
        xlabel="Cross Attention Ratio",
        ylabel="Vision Attention Ratio",
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
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(base_dir, "data", "results", "exp1")
    metrics_csv_path = os.path.join(results_dir, "attention_metrics.csv")
    analysis_dir = os.path.join(results_dir, "analysis")

    records = load_records_from_csv(metrics_csv_path)
    artifacts = generate_analysis_artifacts(records, analysis_dir)

    print("=" * 50)
    print(" 实验一分析完成 ")
    print("=" * 50)
    print(f" 指标文件: {metrics_csv_path}")
    print(f" 摘要文件: {artifacts['summary_path']}")
    print(f" 图表目录: {artifacts['output_dir']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
