import math
import os

import numpy as np


def sanitize_attention_map(attention_map):
    """
    将输入注意力图转换为 float32 numpy 数组，并清理非法值。
    """
    if attention_map is None:
        return None

    attention = np.asarray(attention_map, dtype=np.float32)
    attention = np.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
    return attention


def normalize_attention_for_visualization(attention_map):
    """
    仅用于可视化的归一化，不应直接用于跨样本统计。
    """
    attention = sanitize_attention_map(attention_map)
    if attention is None:
        return None

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
    if bbox is None:
        return None

    image_width, image_height = image_size
    grid_height, grid_width = attention_shape
    if image_width <= 0 or image_height <= 0:
        return None

    left, top, right, bottom = [float(value) for value in bbox]
    left = max(0.0, min(left, image_width))
    right = max(0.0, min(right, image_width))
    top = max(0.0, min(top, image_height))
    bottom = max(0.0, min(bottom, image_height))

    if right <= left or bottom <= top:
        return None

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
            overlap_width = max(
                0.0, min(right, patch_right) - max(left, patch_left)
            )
            if overlap_width <= 0.0:
                continue

            overlap_area = overlap_width * overlap_height
            patch_area = patch_width * patch_height
            overlap_mask[row, col] = overlap_area / patch_area

    return overlap_mask


def compute_attention_metrics(attention_map, bbox, image_size, topk_ratio=0.1):
    """
    基于原始 patch 注意力图，计算文本区域关注程度指标。
    """
    attention = sanitize_attention_map(attention_map)
    if attention is None:
        return {
            "available": False,
            "bbox_attention_sum": None,
            "bbox_attention_ratio": None,
            "bbox_mean_attention": None,
            "bbox_max_attention": None,
            "bbox_patch_coverage": None,
            "topk_patch_in_bbox_ratio": None,
        }

    overlap_mask = create_patch_overlap_mask(bbox, image_size, attention.shape)
    if overlap_mask is None:
        return {
            "available": False,
            "bbox_attention_sum": None,
            "bbox_attention_ratio": None,
            "bbox_mean_attention": None,
            "bbox_max_attention": None,
            "bbox_patch_coverage": None,
            "topk_patch_in_bbox_ratio": None,
        }

    total_attention = float(attention.sum())
    bbox_attention_sum = float((attention * overlap_mask).sum())
    bbox_patch_coverage = float(overlap_mask.sum())
    bbox_mean_attention = (
        bbox_attention_sum / bbox_patch_coverage if bbox_patch_coverage > 0 else None
    )

    overlap_binary = overlap_mask > 0
    bbox_max_attention = (
        float(attention[overlap_binary].max()) if np.any(overlap_binary) else None
    )

    flat_attention = attention.reshape(-1)
    flat_binary_mask = overlap_binary.reshape(-1)
    patch_count = flat_attention.size
    topk_count = max(1, int(math.ceil(patch_count * topk_ratio)))
    topk_indices = np.argsort(flat_attention)[-topk_count:]
    topk_hits = int(flat_binary_mask[topk_indices].sum())
    topk_patch_in_bbox_ratio = topk_hits / topk_count

    return {
        "available": True,
        "bbox_attention_sum": bbox_attention_sum,
        "bbox_attention_ratio": (
            bbox_attention_sum / total_attention if total_attention > 0 else None
        ),
        "bbox_mean_attention": bbox_mean_attention,
        "bbox_max_attention": bbox_max_attention,
        "bbox_patch_coverage": bbox_patch_coverage,
        "topk_patch_in_bbox_ratio": topk_patch_in_bbox_ratio,
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
    bbox_values = [int(round(value)) for value in bbox] if bbox is not None else [None] * 4
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
        record[f"{prefix}_attention_available"] = metrics["available"]
        record[f"{prefix}_bbox_attention_sum"] = metrics["bbox_attention_sum"]
        record[f"{prefix}_bbox_attention_ratio"] = metrics["bbox_attention_ratio"]
        record[f"{prefix}_bbox_mean_attention"] = metrics["bbox_mean_attention"]
        record[f"{prefix}_bbox_max_attention"] = metrics["bbox_max_attention"]
        record[f"{prefix}_bbox_patch_coverage"] = metrics["bbox_patch_coverage"]
        record[f"{prefix}_topk_patch_in_bbox_ratio"] = metrics[
            "topk_patch_in_bbox_ratio"
        ]

    return record


def save_attention_arrays(output_path, bbox, attention_payload):
    """
    将原始注意力图与 bbox 一起保存为 npz，便于后续复现实验分析。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bbox_array = (
        np.asarray(bbox, dtype=np.int32) if bbox is not None else np.asarray([], dtype=np.int32)
    )
    np.savez_compressed(
        output_path,
        bbox=bbox_array,
        cross_attention_raw=sanitize_attention_map(
            attention_payload.get("cross_attention_raw")
        ),
        vision_attention_raw=sanitize_attention_map(
            attention_payload.get("vision_attention_raw")
        ),
    )
