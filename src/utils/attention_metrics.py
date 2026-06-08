import math
import os

import numpy as np


def sanitize_attention_map(attention_map):
    """
    功能描述:
    - 将输入注意力图转换为 `float32` 类型的 `numpy` 数组，并将非法值替换为 0。

    输入参数:
    - attention_map (array-like): 原始注意力图，可以是列表、张量或 `numpy` 数组。

    返回值:
    - numpy.ndarray: 清理后的 `float32` 注意力图。
    """
    attention = np.asarray(attention_map, dtype=np.float32)
    attention = np.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
    return attention


def normalize_attention_for_visualization(attention_map):
    """
    功能描述:
    - 将注意力图线性归一化到 `[0, 1]` 区间，仅用于热力图可视化。

    输入参数:
    - attention_map (array-like): 原始注意力图。

    返回值:
    - numpy.ndarray: 归一化后的 `float32` 注意力图。
    """
    attention = sanitize_attention_map(attention_map)

    min_value = float(attention.min())
    max_value = float(attention.max())
    if math.isclose(max_value, min_value):
        return np.zeros_like(attention, dtype=np.float32)

    return (attention - min_value) / (max_value - min_value)


def create_patch_overlap_mask(bbox, image_size, attention_shape):
    """
    功能描述:
    - 计算文字边界框在注意力 patch 网格上的面积重叠比例掩码。

    输入参数:
    - bbox (tuple[int | float, int | float, int | float, int | float]): 文字区域边界框，
      格式为 `(left, top, right, bottom)`。
    - image_size (tuple[int, int]): 原图尺寸，格式为 `(width, height)`。
    - attention_shape (tuple[int, int]): 注意力图网格尺寸，格式为 `(grid_height, grid_width)`。

    返回值:
    - numpy.ndarray: 与 `attention_shape` 同形状的掩码数组，元素范围为 `[0, 1]`。
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

    # 逐个 patch 计算与 bbox 的面积交集，避免把部分覆盖的 patch 直接当成 0 或 1。
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
    功能描述:
    - 基于原始 patch 注意力图和文字边界框，计算文字区域的关注程度指标。

    输入参数:
    - attention_map (array-like): 原始 patch 级注意力图。
    - bbox (tuple[int | float, int | float, int | float, int | float]): 文字区域边界框。
    - image_size (tuple[int, int]): 原图尺寸，格式为 `(width, height)`。

    返回值:
    - dict[str, float]: 包含 `bbox_attention_sum`、`bbox_attention_ratio`、
      `bbox_mean_attention` 和 `bbox_patch_coverage` 的指标字典。
    """
    attention = sanitize_attention_map(attention_map)
    overlap_mask = create_patch_overlap_mask(bbox, image_size, attention.shape)

    total_attention = float(attention.sum())
    if total_attention <= 0:
        raise ValueError("注意力图总和必须大于 0，无法计算比例指标。")

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
    patch,
    patch_type,
):
    """
    功能描述:
    - 汇总单个样本的基础元数据与两类注意力指标，生成可直接保存的结构化记录。

    输入参数:
    - image_name (str): 图片文件名。
    - status (str): 攻击结果标签，例如 `"SUCCESS"` 或 `"FAIL"`。
    - response (str): 模型生成的响应文本。
    - bbox (tuple[int | float, int | float, int | float, int | float]): 文字区域边界框。
    - image_size (tuple[int, int]): 图像尺寸，格式为 `(width, height)`。
    - cross_attention_raw (array-like): 原始交叉注意力图。
    - vision_attention_raw (array-like): 原始视觉自注意力图。
    - patch (str): patch 实验条件标签。
    - patch_type (str): patch 具体类型名称。

    返回值:
    - dict[str, int | float | str]: 单个样本的结构化分析记录。
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
        "patch": patch,
        "patch_type": patch_type,
        "response": response,
        "image_width": image_width,
        "image_height": image_height,
        "bbox_left": bbox_values[0],
        "bbox_top": bbox_values[1],
        "bbox_right": bbox_values[2],
        "bbox_bottom": bbox_values[3],
    }

    # 两类注意力使用统一前缀写回记录，便于后续直接导出到 CSV/JSON。
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
    功能描述:
    - 将原始注意力图与文字边界框保存为压缩的 `npz` 文件，便于复现实验分析。

    输入参数:
    - output_path (str): 输出文件路径。
    - bbox (tuple[int | float, int | float, int | float, int | float]): 文字区域边界框。
    - attention_payload (dict[str, array-like]): 至少包含 `cross_attention_raw` 与
      `vision_attention_raw` 两个字段的注意力数据字典。

    返回值:
    - None: 函数直接将结果保存到磁盘。
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
