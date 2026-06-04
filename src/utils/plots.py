import matplotlib.pyplot as plt
import numpy as np
import os


def plot_attention_heatmap(image, attention_map, bbox, output_path):
    """
    将注意力热力图叠加在原图上，并画出文字区域的Bounding Box。

    参数:
    - image: PIL Image
    - attention_map: 2D numpy array (与原图等大或可resize的热力图数据), 取值在 [0, 1] 之间。
                     如果是 None，则生成一个随机的高亮区域用于测试流程。
    - bbox: 文本的边界框 (left, top, right, bottom)
    - output_path: 保存路径
    """
    try:
        import cv2
    except ImportError:
        print(
            "警告: 缺少 opencv-python 库，无法绘制热力图，请使用 pip install opencv-python 安装。"
        )
        return

    image_np = np.array(image)

    # 确保 image_np 是 RGB 格式，并且有 3 个通道
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # 如果传入了真实的 attention_map (例如 24x24)，需要 resize 到原图大小
    if attention_map is not None:
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
    if bbox:
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
    plt.title("Attention Heatmap & Text Bounding Box")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    # print(f"Heatmap visualization saved to: {output_path}")
