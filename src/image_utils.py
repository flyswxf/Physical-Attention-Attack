from PIL import Image, ImageDraw, ImageFont
import os
import textwrap


def wrap_text(text, font, max_width):
    """
    根据字体和最大宽度，将长文本自动折行。
    如果文本本身包含 '\n'，也会保留其原本的换行。
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
    在图片中注入文字，模拟物理世界的文本攻击
    返回: 带有文字的PIL Image对象, 以及文字的Bounding Box (left, top, right, bottom)
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 尝试加载字体目录下的字体
    if font_dir:
        font_path = os.path.join(
            font_dir, "TimesNewRoman-BoldItalic_mianfeiziti.com.otf"
        )
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        # print(f"Attack image saved to: {output_path}")

    return image, bbox


def add_attention_patch(image, bbox, patch_type="red_box", output_path=None):
    """
    在文字周围添加一个Patch，以试图吸引模型的注意力 (实验二的预留功能)
    """
    image_with_patch = image.copy()
    draw = ImageDraw.Draw(image_with_patch)
    left, top, right, bottom = bbox

    if patch_type == "red_box":
        # 在文字外围画一个醒目的红色粗框作为 Patch
        padding = 10
        draw.rectangle(
            [left - padding, top - padding, right + padding, bottom + padding],
            outline="red",
            width=5,
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_with_patch.save(output_path)

    return image_with_patch
