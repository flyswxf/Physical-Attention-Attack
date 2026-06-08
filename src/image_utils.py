from PIL import Image, ImageDraw, ImageFont
import os


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


def _load_font(font_dir, font_size):
    """
    功能描述:
    - 根据字体目录和字号加载实验使用的字体；若未提供有效字体文件，则退回默认字体。

    输入参数:
    - font_dir (str | None): 字体目录路径；为 `None` 时直接使用默认字体。
    - font_size (int): 字体大小。

    返回值:
    - PIL.ImageFont.ImageFont: 可直接用于绘制文本的字体对象。
    """
    if not font_dir:
        return ImageFont.load_default()

    font_path = os.path.join(font_dir, "TimesNewRoman-BoldItalic_mianfeiziti.com.otf")
    if not os.path.isfile(font_path):
        return ImageFont.load_default()

    return ImageFont.truetype(font_path, font_size)


def inject_text_to_image(
    image_path,
    text,
    output_path=None,
    position=(50, 50),
    font_size=30,
    color="red",
    font_dir=None,
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
    - font_dir (str | None): 字体目录路径。

    返回值:
    - tuple[PIL.Image.Image, tuple[int, int, int, int]]: 带文字的图像对象，以及文字块
      的边界框 `(left, top, right, bottom)`。
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(font_dir, font_size)

    # 计算文本允许的最大宽度 (图片宽度 - 起始x坐标 - 右侧边距)
    # 假设右侧留白至少为 50 像素
    max_width = image.width - position[0] - 50
    if max_width <= 0:
        max_width = image.width  # fallback

    # 对长文本进行自动换行处理
    wrapped_text = wrap_text(text, font, max_width)

    # 获取多行文字的Bounding Box
    # multiline_textbbox 能够准确计算包含换行符的文本块边界
    bbox = draw.multiline_textbbox(position, wrapped_text, font=font)

    # 将多行文字画在图片上
    draw.multiline_text(position, wrapped_text, fill=color, font=font)

    # 如果指定了输出路径，则保存图片
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)

    return image, bbox


def add_attention_patch(image, bbox, patch_type="red_box", output_path=None):
    """
    功能描述:
    - 在文字边界框周围添加视觉 patch，用于构造额外的注意力干扰条件。

    输入参数:
    - image (PIL.Image.Image): 已注入文字的图像对象。
    - bbox (tuple[int, int, int, int]): 文字区域边界框。
    - patch_type (str): patch 类型，当前支持 `"red_box"`。
    - output_path (str | None): 输出图像路径；为 `None` 时仅返回内存中的图像对象。

    返回值:
    - tuple[PIL.Image.Image, tuple[int, int, int, int]]: 添加 patch 后的图像对象，以及
      patch 的边界框 `(left, top, right, bottom)`。
    """
    image_with_patch = image.copy()
    draw = ImageDraw.Draw(image_with_patch)
    left, top, right, bottom = bbox

    if patch_type == "red_box":
        # 红框直接包围文字区域，确保 patch 的空间位置和注入文本强绑定。
        padding = 10
        patch_bbox = (left - padding, top - padding, right + padding, bottom + padding)
        draw.rectangle(patch_bbox, outline="red", width=5)
    else:
        raise ValueError(f"不支持的 patch_type: {patch_type}")

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image_with_patch.save(output_path)

    return image_with_patch, patch_bbox
