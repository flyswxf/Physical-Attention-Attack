import matplotlib.pyplot as plt
import numpy as np
import os


def plot_attention_heatmap(image, attention_map, bbox, output_path):
    """
    功能描述:
    - 将二维注意力图缩放到原图尺寸后叠加到输入图像上，并标出文字区域边界框。

    输入参数:
    - image (PIL.Image.Image): 原始输入图像。
    - attention_map (numpy.ndarray): 二维注意力矩阵，元素通常位于 `[0, 1]` 区间。
    - bbox (tuple[int | float, int | float, int | float, int | float]): 文字区域边界框，
      格式为 `(left, top, right, bottom)`。
    - output_path (str): 热力图保存路径。

    返回值:
    - None: 函数直接将图像保存到磁盘，不返回额外结果。
    """
    import cv2

    image_np = np.array(image)

    # 确保 image_np 是 RGB 格式，并且有 3 个通道
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    title = "Attention Heatmap & Text Bounding Box"
    # 模型输出通常是 patch 级注意力，这里统一缩放到像素空间后再可视化。
    if (
        attention_map.shape[0] != image_np.shape[0]
        or attention_map.shape[1] != image_np.shape[1]
    ):
        attention_map = cv2.resize(
            attention_map,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    # 归一化并转为伪彩色 (Jet colormap)
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attention_map_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加到原图上 (alpha 是热力图的透明度)
    alpha = 0.5
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    # 画文字的 Bounding Box，方便直观看到模型是否关注了这块区域
    left, top, right, bottom = [int(v) for v in bbox]
    cv2.rectangle(overlay, (left, top), (right, bottom), (255, 0, 0), 3)  # 蓝色框
    cv2.putText(
        overlay,
        "Text Area",
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    # 使用 matplotlib 画出最终的图并保存
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def _collect_group_values(records, metric_key, group_key="status"):
    """
    功能描述:
    - 按给定分组字段收集指定指标的数值列表，供箱线图等统计图复用。

    输入参数:
    - records (list[dict]): 结构化实验记录列表。
    - metric_key (str): 需要提取的指标字段名。
    - group_key (str): 用于分组的字段名，默认值为 `"status"`。

    返回值:
    - dict[str, list[float]]: 以分组名为键、指标数值列表为值的字典。
    """
    grouped_values = {}
    for record in records:
        group_name = record.get(group_key, "UNKNOWN")
        grouped_values.setdefault(group_name, []).append(float(record[metric_key]))
    return grouped_values


def plot_metric_boxplot(
    records, metric_key, output_path, title, ylabel, group_key="status"
):
    """
    功能描述:
    - 将指定指标按分组字段绘制为箱线图，用于比较不同组别的分布差异。

    输入参数:
    - records (list[dict]): 结构化实验记录列表。
    - metric_key (str): 需要绘制的指标字段名。
    - output_path (str): 图像保存路径。
    - title (str): 图表标题。
    - ylabel (str): 纵轴标签。
    - group_key (str): 分组字段名，默认值为 `"status"`。

    返回值:
    - None: 函数直接将图表写入磁盘。
    """
    grouped_values = _collect_group_values(records, metric_key, group_key=group_key)
    if not grouped_values:
        return

    labels = list(grouped_values.keys())
    data = [grouped_values[label] for label in labels]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metric_scatter(
    records,
    x_key,
    y_key,
    output_path,
    title,
    xlabel,
    ylabel,
    group_key="status",
):
    """
    功能描述:
    - 绘制两个指标之间的散点关系图，并按分组字段区分颜色。

    输入参数:
    - records (list[dict]): 结构化实验记录列表。
    - x_key (str): 横轴指标字段名。
    - y_key (str): 纵轴指标字段名。
    - output_path (str): 图像保存路径。
    - title (str): 图表标题。
    - xlabel (str): 横轴标签。
    - ylabel (str): 纵轴标签。
    - group_key (str): 分组字段名，默认值为 `"status"`。

    返回值:
    - None: 函数直接将图表写入磁盘。
    """
    group_names = sorted({record.get(group_key, "UNKNOWN") for record in records})
    color_values = plt.cm.tab10(np.linspace(0, 1, max(len(group_names), 1)))
    colors = {
        group_name: color_values[index] for index, group_name in enumerate(group_names)
    }

    plt.figure(figsize=(8, 6))
    plotted_groups = set()
    for record in records:
        group_name = record.get(group_key, "UNKNOWN")
        color = colors.get(group_name, "#1f77b4")
        label = group_name if group_name not in plotted_groups else None
        plotted_groups.add(group_name)
        plt.scatter(
            float(record[x_key]),
            float(record[y_key]),
            alpha=0.7,
            color=color,
            label=label,
        )

    if plotted_groups:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle="--", alpha=0.3)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
