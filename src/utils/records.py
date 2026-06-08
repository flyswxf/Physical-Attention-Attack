import csv
import os

REGION_NAMES = ("bbox", "patch_bbox", "expanded_bbox")
ATTENTION_PREFIXES = ("cross", "vision")
ATTENTION_SUFFIXES = (
    "attention_sum",
    "attention_ratio",
    "mean_attention",
    "patch_coverage",
)


def _build_numeric_fields():
    """
    功能描述:
    - 生成结构化记录中所有需要按浮点数恢复的字段名集合。

    输入参数:
    - 无。

    返回值:
    - set[str]: 数值字段名称集合。
    """
    numeric_fields = {
        "image_width",
        "image_height",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "patch_bbox_left",
        "patch_bbox_top",
        "patch_bbox_right",
        "patch_bbox_bottom",
        "expanded_bbox_left",
        "expanded_bbox_top",
        "expanded_bbox_right",
        "expanded_bbox_bottom",
    }

    for attention_prefix in ATTENTION_PREFIXES:
        for region_name in REGION_NAMES:
            for attention_suffix in ATTENTION_SUFFIXES:
                numeric_fields.add(
                    f"{attention_prefix}_{region_name}_{attention_suffix}"
                )

    return numeric_fields


NUMERIC_FIELDS = _build_numeric_fields()


def save_records_to_csv(records, output_path):
    """
    功能描述:
    - 将结构化实验记录保存为 CSV 文件，供后续分析脚本直接读取。

    输入参数:
    - records (list[dict]): 待保存的结构化实验记录列表。
    - output_path (str): CSV 输出路径。

    返回值:
    - None: 函数直接将结果写入磁盘。
    """
    if not records:
        raise ValueError("records 不能为空，无法写入 CSV。")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = list(records[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_records_from_csv(csv_path):
    """
    功能描述:
    - 从 CSV 文件中读取实验记录，并将数值字段恢复为浮点数。

    输入参数:
    - csv_path (str): CSV 文件路径。

    返回值:
    - list[dict]: 解析后的结构化实验记录列表。
    """
    records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            parsed_row = {}
            for key, value in row.items():
                if value == "":
                    raise ValueError(f"CSV 字段为空: key={key}")
                # CSV 读入后全部是字符串，这里按字段表恢复基础数值类型。
                if key in NUMERIC_FIELDS:
                    parsed_row[key] = float(value)
                else:
                    parsed_row[key] = value
            records.append(parsed_row)
    return records
