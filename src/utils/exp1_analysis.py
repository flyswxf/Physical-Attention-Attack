import csv
import json
import os
from statistics import mean, median

from .plots import plot_metric_boxplot, plot_metric_scatter


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
    "cross_bbox_max_attention",
    "cross_bbox_patch_coverage",
    "cross_topk_patch_in_bbox_ratio",
    "vision_bbox_attention_sum",
    "vision_bbox_attention_ratio",
    "vision_bbox_mean_attention",
    "vision_bbox_max_attention",
    "vision_bbox_patch_coverage",
    "vision_topk_patch_in_bbox_ratio",
}

BOOLEAN_FIELDS = {
    "cross_attention_available",
    "vision_attention_available",
}


def save_records_to_csv(records, output_path):
    """
    将结构化实验记录保存为 CSV。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_records_from_csv(csv_path):
    """
    从 CSV 读取实验记录，并恢复基础类型。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到实验指标文件: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            parsed_row = {}
            for key, value in row.items():
                if value == "":
                    parsed_row[key] = None
                elif key in BOOLEAN_FIELDS:
                    parsed_row[key] = value.lower() == "true"
                elif key in NUMERIC_FIELDS:
                    parsed_row[key] = float(value)
                else:
                    parsed_row[key] = value
            records.append(parsed_row)
    return records


def _collect_metric_values(records, metric_key, status):
    values = []
    for record in records:
        if record.get("status") != status:
            continue
        value = record.get(metric_key)
        if value is not None:
            values.append(float(value))
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
        "cross_topk_patch_in_bbox_ratio",
        "vision_bbox_attention_ratio",
        "vision_bbox_mean_attention",
        "vision_topk_patch_in_bbox_ratio",
    ]
    for metric_key in metric_keys:
        summary["metrics"][metric_key] = {}
        for status in statuses:
            values = _collect_metric_values(records, metric_key, status)
            if not values:
                summary["metrics"][metric_key][status] = {
                    "count": 0,
                    "mean": None,
                    "median": None,
                }
                continue

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
    plot_metric_boxplot(
        records,
        "cross_topk_patch_in_bbox_ratio",
        os.path.join(output_dir, "cross_topk_ratio_boxplot.png"),
        title="Cross Attention Top-K Coverage",
        ylabel="Top-K Patch Ratio",
    )
    plot_metric_boxplot(
        records,
        "vision_topk_patch_in_bbox_ratio",
        os.path.join(output_dir, "vision_topk_ratio_boxplot.png"),
        title="Vision Attention Top-K Coverage",
        ylabel="Top-K Patch Ratio",
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
