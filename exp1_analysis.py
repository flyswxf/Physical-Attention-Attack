import argparse
import json
import os
from statistics import mean, median

from src.utils.plots import plot_metric_boxplot
from src.utils.records import REGION_NAMES, load_records_from_csv

REGION_LABELS = {
    "bbox": "Text BBox",
    "patch_bbox": "Patch BBox",
    "expanded_bbox": "Expanded BBox",
}
ATTENTION_TYPE_LABELS = {
    "cross": "Cross Attention",
    "vision": "Vision Attention",
}
METRIC_TYPE_LABELS = {
    "attention_ratio": "Attention Ratio",
    "mean_attention": "Mean Attention",
}


def _build_metric_specs():
    """
    功能描述:
    - 生成需要参与摘要统计和绘图的指标规格列表。

    输入参数:
    - 无。

    返回值:
    - list[dict[str, str]]: 每个元素描述一个指标字段及其元信息。
    """
    metric_specs = []
    for region_name in REGION_NAMES:
        for attention_type in ("cross", "vision"):
            for metric_type in ("attention_ratio", "mean_attention"):
                metric_specs.append(
                    {
                        "key": f"{attention_type}_{region_name}_{metric_type}",
                        "region_name": region_name,
                        "attention_type": attention_type,
                        "metric_type": metric_type,
                    }
                )
    return metric_specs


def _collect_metric_values(records, metric_key, group_key, group_value):
    """
    功能描述:
    - 从结构化记录中筛选指定分组对应的某个指标值列表。

    输入参数:
    - records (list[dict]): 结构化实验记录列表。
    - metric_key (str): 目标指标字段名。
    - group_key (str): 分组字段名。
    - group_value (str): 当前需要筛选的分组取值。

    返回值:
    - list[float]: 当前分组下该指标的数值列表。
    """
    values = []
    for record in records:
        if record.get(group_key) != group_value:
            continue
        values.append(float(record[metric_key]))
    return values


def summarize_records(records, group_key="status"):
    """
    功能描述:
    - 按给定分组维度统计核心注意力指标，并生成均值、中位数等摘要信息。

    输入参数:
    - records (list[dict]): 结构化实验记录列表。
    - group_key (str): 分组字段名，默认值为 `"status"`。

    返回值:
    - dict: 按分组组织的样本数与指标摘要结果。
    """
    if not records:
        raise ValueError("records 不能为空。")
    if group_key not in records[0]:
        raise KeyError(f"记录中不存在分组字段: {group_key}")

    summary = {
        "sample_count": len(records),
        "group_key": group_key,
        "group_counts": {},
        "regions": {},
    }
    group_values = sorted({record[group_key] for record in records})
    for group_value in group_values:
        summary["group_counts"][group_value] = sum(
            1 for record in records if record[group_key] == group_value
        )

    metric_specs = _build_metric_specs()
    for region_name in REGION_NAMES:
        summary["regions"][region_name] = {
            "label": REGION_LABELS[region_name],
            "metrics": {},
        }

    for metric_spec in metric_specs:
        metric_key = metric_spec["key"]
        region_name = metric_spec["region_name"]
        attention_type = metric_spec["attention_type"]
        metric_type = metric_spec["metric_type"]
        grouped_metric_name = f"{attention_type}_{metric_type}"
        summary["regions"][region_name]["metrics"][grouped_metric_name] = {
            "label": (
                f"{ATTENTION_TYPE_LABELS[attention_type]} "
                f"{METRIC_TYPE_LABELS[metric_type]}"
            ),
            "field": metric_key,
            "groups": {},
        }
        for group_value in group_values:
            # 每个分组都要求有完整的指标值，避免后续均值和中位数统计失真。
            values = _collect_metric_values(records, metric_key, group_key, group_value)
            if not values:
                raise ValueError(
                    f"指标缺失: metric_key={metric_key}, "
                    f"group_key={group_key}, group_value={group_value}"
                )

            summary["regions"][region_name]["metrics"][grouped_metric_name]["groups"][
                group_value
            ] = {
                "count": len(values),
                "mean": mean(values),
                "median": median(values),
            }

    return summary


def generate_analysis_artifacts(records, output_dir, group_key="status"):
    """
    功能描述:
    - 基于结构化实验记录生成 JSON 摘要和分组箱线图。

    输入参数:
    - records (list[dict]): 结构化实验记录列表。
    - output_dir (str): 分析产物输出目录。
    - group_key (str): 分组字段名，例如 `"status"` 或 `"patch"`。

    返回值:
    - dict[str, str]: 包含摘要文件路径和输出目录的结果字典。
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = summarize_records(records, group_key=group_key)

    summary_path = os.path.join(output_dir, f"attention_summary_by_{group_key}.json")
    with open(summary_path, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=4)

    for region_name in REGION_NAMES:
        region_label = REGION_LABELS[region_name]
        region_output_dir = os.path.join(output_dir, region_name)
        os.makedirs(region_output_dir, exist_ok=True)
        plot_metric_boxplot(
            records,
            f"cross_{region_name}_attention_ratio",
            os.path.join(
                region_output_dir, f"cross_attention_ratio_by_{group_key}.png"
            ),
            title=f"{region_label}: Cross Attention Ratio by {group_key}",
            ylabel="Attention Ratio",
            group_key=group_key,
        )
        plot_metric_boxplot(
            records,
            f"vision_{region_name}_attention_ratio",
            os.path.join(
                region_output_dir,
                f"vision_attention_ratio_by_{group_key}.png",
            ),
            title=f"{region_label}: Vision Attention Ratio by {group_key}",
            ylabel="Attention Ratio",
            group_key=group_key,
        )
        plot_metric_boxplot(
            records,
            f"cross_{region_name}_mean_attention",
            os.path.join(region_output_dir, f"cross_mean_attention_by_{group_key}.png"),
            title=f"{region_label}: Cross Mean Attention by {group_key}",
            ylabel="Attention Strength",
            group_key=group_key,
        )
        plot_metric_boxplot(
            records,
            f"vision_{region_name}_mean_attention",
            os.path.join(
                region_output_dir, f"vision_mean_attention_by_{group_key}.png"
            ),
            title=f"{region_label}: Vision Mean Attention by {group_key}",
            ylabel="Attention Strength",
            group_key=group_key,
        )

    return {
        "summary_path": summary_path,
        "output_dir": output_dir,
    }


def parse_args():
    """
    功能描述:
    - 解析实验一分析脚本的命令行参数。

    输入参数:
    - 无。

    返回值:
    - argparse.Namespace: 解析后的命令行参数对象。
    """
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
    """
    功能描述:
    - 读取实验一生成的结构化记录，并输出指定分组维度下的摘要与图表。

    输入参数:
    - 无。

    返回值:
    - None: 函数执行完成后直接打印结果路径。
    """
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
