from PIL import Image, ImageDraw, ImageFont
import os


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
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # 获取文字的Bounding Box
    bbox = draw.textbbox(position, text, font=font)

    # 将文字画在图片上
    draw.text(position, text, fill=color, font=font)

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
