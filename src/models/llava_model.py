import torch
import numpy as np
from .base_model import BaseModel


class LLaVAModel(BaseModel):
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        super().__init__(model_id, device)

    def load_model(self):
        """
        加载LLaVA模型和处理器 (如果本地显存不够，可改用较小的模型或量化加载)
        由于这里需要提取注意力，所以加载完整的HuggingFace模型。
        """
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError:
            print("警告: 缺少 transformers 库，请使用 pip install transformers 安装。")
            return False

        print(f"Loading {self.model_id} (Multi-GPU auto mode)...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # 为了提取attention，保持默认或者用float16。
        # 注意：这里需要设置 attn_implementation="eager"，否则会报 sdpa 错误
        # 使用 device_map="auto" 让 transformers 自动分配模型层到两个 T4 GPU 上
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto",
        )

        # 记录模型实际分配的设备（可能不再单纯是 'cuda:0'）
        self.device = self.model.device

        return True

    def run_inference_and_get_attention(self, image, prompt):
        """
        运行推理并提取注意力权重
        """
        if self.model is None or self.processor is None:
            print("模型未加载，返回模拟结果...")
            return "This is a simulated response due to model not being loaded.", None

        # LLaVA的对话格式
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=formatted_prompt, images=image, return_tensors="pt"
        ).to(self.device, torch.float16)

        with torch.no_grad():
            # 设置 output_attentions=True 获取注意力图
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # 1. 解码生成的文本
        generated_text = self.processor.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        # 2. 提取 Attention Maps
        # out.attentions：tuple[layer_num], 单层shape=[B, num_head, q_len, all_seq_len]
        # LLaVA 的输出 output.attentions 是一个元组，长度为生成的 token 数量。
        # 每一步的 attention 是一个元组，包含各层的 attention 张量。
        attention_map_2d = None
        if hasattr(outputs, "attentions"):
            try:
                # LLaVA 1.5 默认将图片编码为 576 个 patches (24x24)
                num_image_tokens = 576
                image_token_id = 32000  # LLaVA 中的 <image> token ID

                input_ids = inputs["input_ids"][0]
                image_idx = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                # 找到图片 token 开始的位置
                image_start_idx = image_idx[0].item() if len(image_idx) > 0 else 3

                all_image_attn = []
                # 遍历每一步生成的 token 的 attention
                for step_idx, step_attn in enumerate(outputs.attentions):
                    # 取最后一层的 attention: shape (batch, heads, q_len, k_len)
                    last_layer_attn = step_attn[-1]
                    # 在所有的注意力头上取平均: shape (q_len, k_len)
                    attn = last_layer_attn.mean(dim=1).squeeze(0)

                    if step_idx == 0:
                        # 第一步，q_len 是整个 prompt 的长度。预测第一个词的是 prompt 的最后一个 token
                        token_attn = attn[-1, :]
                    else:
                        # 后续步，q_len 是 1。当前生成的 token 对前面所有 token 的 attention
                        token_attn = attn[0, :]

                    # 截取对图片 patches 的 attention
                    img_attn = token_attn[
                        image_start_idx : image_start_idx + num_image_tokens
                    ]
                    all_image_attn.append(img_attn)

                if all_image_attn:
                    # 将所有生成 token 对图片的 attention 叠加并取平均
                    avg_image_attn = torch.stack(all_image_attn).mean(dim=0)

                    if len(avg_image_attn) == num_image_tokens:
                        # 转换成 24x24 的 2D Numpy 数组
                        attention_map_2d = (
                            avg_image_attn.reshape(24, 24)
                            .cpu()
                            .to(torch.float32)
                            .numpy()
                        )
                        # 对稀疏的注意力进行对数缩放，增强微弱注意力的可视化效果
                        epsilon = 1e-6
                        attention_map_2d = np.log(attention_map_2d + epsilon)
                        # 归一化到 [0, 1] 区间以便后续绘图
                        attention_map_2d = (
                            attention_map_2d - attention_map_2d.min()
                        ) / (attention_map_2d.max() - attention_map_2d.min() + 1e-8)
                    else:
                        print(
                            f"警告: 提取的图像注意力长度 ({len(avg_image_attn)}) 与预期的 ({num_image_tokens}) 不符。"
                        )

                    # 清理中间张量以释放显存
                    del all_image_attn
                    del avg_image_attn
            except Exception as e:
                print(f"注意力提取解析失败: {e}")

        # 3. 提取 Vision Tower 的自注意力 (视觉模块最关注的地方)
        vision_attention_map_2d = None
        if "pixel_values" in inputs:
            try:
                with torch.no_grad():
                    # 显式调用 vision tower 提取注意力
                    vision_outputs = self.model.vision_tower(
                        inputs["pixel_values"], output_attentions=True
                    )
                if (
                    hasattr(vision_outputs, "attentions")
                    and vision_outputs.attentions is not None
                ):
                    # 取最后一层的 attention: shape (batch, heads, seq_len, seq_len)
                    last_layer_vision_attn = vision_outputs.attentions[-1]
                    # 在所有的注意力头上取平均: shape (seq_len, seq_len)
                    avg_vision_attn = last_layer_vision_attn.mean(dim=1).squeeze(0)

                    # CLIP ViT 的第一个 token 是 CLS token
                    # 提取 CLS token 对图像 patches (索引 1 到最后) 的注意力
                    cls_attn = avg_vision_attn[0, 1:]

                    if len(cls_attn) == 576:
                        vision_attention_map_2d = (
                            cls_attn.reshape(24, 24).cpu().to(torch.float32).numpy()
                        )
                        vision_attention_map_2d = (
                            vision_attention_map_2d - vision_attention_map_2d.min()
                        ) / (
                            vision_attention_map_2d.max()
                            - vision_attention_map_2d.min()
                            + 1e-8
                        )
                    else:
                        print(
                            f"警告: 视觉模块注意力长度 ({len(cls_attn)}) 与预期的 (576) 不符。"
                        )

                    del last_layer_vision_attn
                    del avg_vision_attn
            except Exception as e:
                print(f"视觉模块注意力提取解析失败: {e}")

        # 将两种注意力打包返回
        attention_dict = {
            "cross_attention": attention_map_2d,
            "vision_attention": vision_attention_map_2d,
        }

        return generated_text, attention_dict
