import csv
import os

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
