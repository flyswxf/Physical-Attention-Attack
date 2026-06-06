import math
import os

import numpy as np


def sanitize_attention_map(attention_map):
    """
    将输入注意力图转换为 float32 numpy 数组，并清理非法值。
    """

    attention = np.asarray(attention_map, dtype=np.float32)
    attention = np.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
    return attention


def normalize_attention_for_visualization(attention_map):
    """
    仅用于可视化的归一化，不应直接用于跨样本统计。
    """
    attention = sanitize_attention_map(attention_map)

    min_value = float(attention.min())
    max_value = float(attention.max())
    if math.isclose(max_value, min_value):
        return np.zeros_like(attention, dtype=np.float32)

    return (attention - min_value) / (max_value - min_value)


def create_patch_overlap_mask(bbox, image_size, attention_shape):
    """
    计算 bbox 在注意力 patch 网格上的面积重叠比例。

    返回:
    - overlap_mask: shape 与 attention_shape 相同，元素取值为 [0, 1]
    """
    image_width, image_height = image_size
    grid_height, grid_width = attention_shape

    left, top, right, bottom = [float(value) for value in bbox]
    left = max(0.0, min(left, image_width))
    right = max(0.0, min(right, image_width))
    top = max(0.0, min(top, image_height))
    bottom = max(0.0, min(bottom, image_height))

    if right <= left or bottom <= top:
        raise ValueError(f"非法 bbox: {bbox}")

    patch_width = image_width / grid_width
    patch_height = image_height / grid_height
    overlap_mask = np.zeros((grid_height, grid_width), dtype=np.float32)

    for row in range(grid_height):
        patch_top = row * patch_height
        patch_bottom = (row + 1) * patch_height
        overlap_height = max(0.0, min(bottom, patch_bottom) - max(top, patch_top))
        if overlap_height <= 0.0:
            continue

        for col in range(grid_width):
            patch_left = col * patch_width
            patch_right = (col + 1) * patch_width
            overlap_width = max(0.0, min(right, patch_right) - max(left, patch_left))
            if overlap_width <= 0.0:
                continue

            overlap_area = overlap_width * overlap_height
            patch_area = patch_width * patch_height
            overlap_mask[row, col] = overlap_area / patch_area

    return overlap_mask


def compute_attention_metrics(attention_map, bbox, image_size):
    """
    基于原始 patch 注意力图，计算文本区域关注程度指标。

    返回字段说明:
    - bbox_attention_sum: 文本框区域内的加权注意力总量。由于使用了 bbox 与 patch
      的面积重叠比例，这个值表示“落在文字区域里的总注意力质量”。
    - bbox_attention_ratio: 文本框区域内注意力总量占整张图总注意力的比例。
      这是最适合跨样本比较的核心指标之一，表示模型有多少注意力分配给了文字区域。
    - bbox_mean_attention: 文本框覆盖到的 patch 区域上的平均注意力强度。
      它反映的是文字区域内部“平均每单位 patch 面积”获得了多少注意力。
    - bbox_patch_coverage: 文本框在 patch 网格上覆盖的总 patch 面积，
      这里是按重叠比例累计后的结果，不一定是整数。
    """
    attention = sanitize_attention_map(attention_map)
    overlap_mask = create_patch_overlap_mask(bbox, image_size, attention.shape)

    total_attention = float(attention.sum())
    bbox_attention_sum = float((attention * overlap_mask).sum())
    bbox_patch_coverage = float(overlap_mask.sum())
    if bbox_patch_coverage <= 0:
        raise ValueError("bbox 与注意力 patch 网格没有重叠区域。")

    bbox_mean_attention = bbox_attention_sum / bbox_patch_coverage

    overlap_binary = overlap_mask > 0
    if not np.any(overlap_binary):
        raise ValueError("bbox 未覆盖任何注意力 patch。")

    return {
        "bbox_attention_sum": bbox_attention_sum,
        "bbox_attention_ratio": bbox_attention_sum / total_attention,
        "bbox_mean_attention": bbox_mean_attention,
        "bbox_patch_coverage": bbox_patch_coverage,
    }


def build_attention_record(
    image_name,
    status,
    response,
    bbox,
    image_size,
    cross_attention_raw,
    vision_attention_raw,
):
    """
    组装单个样本的结构化分析记录。
    """
    bbox_values = [int(round(value)) for value in bbox]
    image_width, image_height = image_size

    cross_metrics = compute_attention_metrics(
        cross_attention_raw,
        bbox,
        image_size,
    )
    vision_metrics = compute_attention_metrics(
        vision_attention_raw,
        bbox,
        image_size,
    )

    record = {
        "image": image_name,
        "status": status,
        "response": response,
        "image_width": image_width,
        "image_height": image_height,
        "bbox_left": bbox_values[0],
        "bbox_top": bbox_values[1],
        "bbox_right": bbox_values[2],
        "bbox_bottom": bbox_values[3],
    }

    for prefix, metrics in (
        ("cross", cross_metrics),
        ("vision", vision_metrics),
    ):
        record[f"{prefix}_bbox_attention_sum"] = metrics["bbox_attention_sum"]
        record[f"{prefix}_bbox_attention_ratio"] = metrics["bbox_attention_ratio"]
        record[f"{prefix}_bbox_mean_attention"] = metrics["bbox_mean_attention"]
        record[f"{prefix}_bbox_patch_coverage"] = metrics["bbox_patch_coverage"]

    return record


def save_attention_arrays(output_path, bbox, attention_payload):
    """
    将原始注意力图与 bbox 一起保存为 npz，便于后续复现实验分析。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bbox_array = np.asarray(bbox, dtype=np.int32)
    np.savez_compressed(
        output_path,
        bbox=bbox_array,
        cross_attention_raw=sanitize_attention_map(
            attention_payload["cross_attention_raw"]
        ),
        vision_attention_raw=sanitize_attention_map(
            attention_payload["vision_attention_raw"]
        ),
    )
