from PIL import Image, ImageDraw, ImageFont
import os


DEFAULT_TEXT_RIGHT_MARGIN = 50


def expand_bbox(bbox, padding, image_size=None):
    """
    功能描述:
    - 将输入边界框向四周扩展指定像素，并可选地裁剪到图像边界内。

    输入参数:
    - bbox (tuple[int, int, int, int]): 原始边界框 `(left, top, right, bottom)`。
    - padding (int): 向四周扩展的像素数。
    - image_size (tuple[int, int] | None): 图像尺寸 `(width, height)`；为 `None` 时不裁剪。

    返回值:
    - tuple[int, int, int, int]: 扩展后的边界框。
    """
    left, top, right, bottom = [int(round(value)) for value in bbox]
    expanded_bbox = (
        left - padding,
        top - padding,
        right + padding,
        bottom + padding,
    )

    if image_size is None:
        return expanded_bbox

    image_width, image_height = image_size
    return (
        max(0, expanded_bbox[0]),
        max(0, expanded_bbox[1]),
        min(image_width, expanded_bbox[2]),
        min(image_height, expanded_bbox[3]),
    )


def wrap_text(text, font, max_width):
    """
    功能描述:
    - 根据字体宽度和最大像素宽度对文本进行自动折行，并保留原始换行结构。

    输入参数:
    - text (str): 待绘制的文本内容。
    - font (PIL.ImageFont.ImageFont): 用于测量文本宽度的字体对象。
    - max_width (int): 每行允许的最大像素宽度。

    返回值:
    - str: 处理后的多行文本字符串。
    """
    lines = []
    # 首先处理用户自己传入的换行符
    paragraphs = text.split("\n")

    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split(" ")
        current_line = []

        for word in words:
            current_line.append(word)
            # 计算当前行的宽度
            test_line = " ".join(current_line)
            # 在较高版本的 PIL 中，getsize 被废弃，应使用 getbbox
            # bbox = (left, top, right, bottom)
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]

            if line_width > max_width:
                if len(current_line) == 1:
                    # 单个单词就超过了宽度，没办法，只能硬着头皮加进去
                    lines.append(current_line[0])
                    current_line = []
                else:
                    # 弹出最后一个单词，把前面的作为一行
                    current_line.pop()
                    lines.append(" ".join(current_line))
                    current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

    return "\n".join(lines)


def load_font(font_dir, font_size):
    """
    功能描述:
    - 从固定字体文件路径加载实验使用的字体；若字体文件不存在，则直接报错退出。

    输入参数:
    - font_dir (str): 字体目录路径。
    - font_size (int): 字体大小。

    返回值:
    - PIL.ImageFont.ImageFont: 可直接用于绘制文本的字体对象。
    """
    if not font_dir:
        raise ValueError("font_dir 不能为空，且必须指向包含目标字体文件的目录。")
    font_path = os.path.join(font_dir, "TimesNewRoman-BoldItalic_mianfeiziti.com.otf")
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    return ImageFont.truetype(font_path, font_size)


def _compute_text_layout(draw, text, position, font, max_width):
    """
    功能描述:
    - 基于给定字体与最大宽度，计算自动换行后的文本及其边界框。

    输入参数:
    - draw (PIL.ImageDraw.ImageDraw): 当前图像的绘制对象。
    - text (str): 待绘制文本。
    - position (tuple[int, int]): 文本绘制起点坐标。
    - font (PIL.ImageFont.ImageFont): 当前字体对象。
    - max_width (int): 文本允许占用的最大像素宽度。

    返回值:
    - tuple[str, tuple[int, int, int, int]]: 自动换行后的文本，以及对应边界框
      `(left, top, right, bottom)`。
    """
    wrapped_text = wrap_text(text, font, max_width)
    bbox = draw.multiline_textbbox(position, wrapped_text, font=font)
    return wrapped_text, bbox


def _bbox_area(bbox):
    """
    功能描述:
    - 计算边界框面积。

    输入参数:
    - bbox (tuple[int, int, int, int]): 边界框 `(left, top, right, bottom)`。

    返回值:
    - int: 边界框面积。
    """
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _fit_font_size_by_area_ratio(
    draw,
    image,
    text,
    position,
    font_dir,
    text_area_ratio,
    max_width,
):
    """
    功能描述:
    - 根据目标面积占比自动搜索最接近的字号，并返回对应的文本布局结果。

    输入参数:
    - draw (PIL.ImageDraw.ImageDraw): 当前图像的绘制对象。
    - image (PIL.Image.Image): 当前图像对象。
    - text (str): 待绘制文本。
    - position (tuple[int, int]): 文本绘制起点坐标。
    - font_dir (str): 字体目录路径，必须包含目标字体文件。
    - text_area_ratio (float): 目标文字边界框面积占图像面积的比例，范围 `(0, 1]`。
    - max_width (int): 文本允许占用的最大像素宽度。

    返回值:
    - tuple[PIL.ImageFont.FreeTypeFont, str, tuple[int, int, int, int]]: 最终字体对象、
      自动换行后的文本，以及对应边界框。
    """
    if not 0 < text_area_ratio <= 1:
        raise ValueError(
            f"text_area_ratio 必须在 (0, 1] 范围内，当前值为: {text_area_ratio}"
        )

    image_area = image.width * image.height
    target_area = image_area * text_area_ratio
    min_font_size = 1
    max_font_size = max(image.width, image.height)

    best_font = None
    best_text = None
    best_bbox = None
    best_diff = None

    left = min_font_size
    right = max_font_size
    # 文本面积随字号整体近似单调增大，因此用二分搜索快速逼近目标占比。
    while left <= right:
        mid = (left + right) // 2
        font = load_font(font_dir, mid)
        wrapped_text, bbox = _compute_text_layout(draw, text, position, font, max_width)
        current_area = _bbox_area(bbox)
        current_diff = abs(current_area - target_area)

        if best_diff is None or current_diff < best_diff:
            best_font = font
            best_text = wrapped_text
            best_bbox = bbox
            best_diff = current_diff

        if current_area < target_area:
            left = mid + 1
        else:
            right = mid - 1

    return best_font, best_text, best_bbox


def inject_text_to_image(
    image_path,
    text,
    output_path=None,
    position=(50, 50),
    font_size=30,
    color="red",
    font_dir=None,
    text_area_ratio=None,
):
    """
    功能描述:
    - 在输入图像上绘制攻击文本，并返回带文字的图像与文字区域边界框。

    输入参数:
    - image_path (str): 原始图像路径。
    - text (str): 需要注入的攻击文本。
    - output_path (str | None): 输出图像路径；为 `None` 时仅返回内存中的图像对象。
    - position (tuple[int, int]): 文本绘制起点坐标，格式为 `(x, y)`。
    - font_size (int): 文本字号。
    - color (str | tuple[int, int, int]): 文本颜色。
    - font_dir (str): 字体目录路径，必须包含目标字体文件。
    - text_area_ratio (float | None): 目标文字边界框面积占图像面积的比例；为 `None`
      时使用固定 `font_size`，否则自动搜索最接近该占比的字号。

    返回值:
    - tuple[PIL.Image.Image, tuple[int, int, int, int]]: 带文字的图像对象，以及文字块
      的边界框 `(left, top, right, bottom)`。
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(font_dir, font_size)

    # 计算文本允许的最大宽度 (图片宽度 - 起始x坐标 - 右侧边距)
    # 假设右侧留白至少为 50 像素
    max_width = image.width - position[0] - DEFAULT_TEXT_RIGHT_MARGIN
    if max_width <= 0:
        max_width = image.width  # fallback

    if text_area_ratio is not None:
        font, wrapped_text, bbox = _fit_font_size_by_area_ratio(
            draw=draw,
            image=image,
            text=text,
            position=position,
            font_dir=font_dir,
            text_area_ratio=text_area_ratio,
            max_width=max_width,
        )
    else:
        wrapped_text, bbox = _compute_text_layout(
            draw=draw,
            text=text,
            position=position,
            font=font,
            max_width=max_width,
        )

    # 将多行文字画在图片上
    draw.multiline_text(position, wrapped_text, fill=color, font=font)

    # 如果指定了输出路径，则保存图片
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)

    return image, bbox


def add_attention_patch(
    image, bbox, patch_type="red_box", output_path=None, padding=10
):
    """
    功能描述:
    - 在文字边界框周围添加视觉 patch，用于构造额外的注意力干扰条件。

    输入参数:
    - image (PIL.Image.Image): 已注入文字的图像对象。
    - bbox (tuple[int, int, int, int]): 文字区域边界框。
    - patch_type (str): patch 类型，当前支持 `"red_box"`。
    - output_path (str | None): 输出图像路径；为 `None` 时仅返回内存中的图像对象。
    - padding (int): patch 相对文字框向外扩展的像素数。

    返回值:
    - tuple[PIL.Image.Image, tuple[int, int, int, int]]: 添加 patch 后的图像对象，以及
      patch 的边界框 `(left, top, right, bottom)`。
    """
    image_with_patch = image.copy()
    draw = ImageDraw.Draw(image_with_patch)
    if patch_type == "red_box":
        # 红框直接包围文字区域，确保 patch 的空间位置和注入文本强绑定。
        patch_bbox = expand_bbox(bbox, padding, image_with_patch.size)
        draw.rectangle(patch_bbox, outline="red", width=5)
    else:
        raise ValueError(f"不支持的 patch_type: {patch_type}")

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image_with_patch.save(output_path)

    return image_with_patch, patch_bbox
