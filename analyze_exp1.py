import argparse
import os

from src.utils.exp1_analysis import generate_analysis_artifacts, load_records_from_csv


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
