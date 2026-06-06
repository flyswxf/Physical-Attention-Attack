import torch
import numpy as np
from .base_model import BaseModel
from ..utils.attention_metrics import normalize_attention_for_visualization


class LLaVAModel(BaseModel):
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        super().__init__(model_id, device)

    def load_model(self):
        """
        加载LLaVA模型和处理器 (如果本地显存不够，可改用较小的模型或量化加载)
        由于这里需要提取注意力，所以加载完整的HuggingFace模型。
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

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
            raise RuntimeError("模型或处理器未加载，无法执行推理。")

        # LLaVA的对话格式
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=formatted_prompt, images=image, return_tensors="pt"
        ).to(self.device, torch.float16)

        with torch.no_grad():
            # 设置 output_attentions=True 获取注意力图
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # 1. 解码生成的文本
        generated_text = self.processor.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        # 2. 提取 Cross Attention Maps
        # out.attentions：tuple[layer_num], 单层shape=[B, num_head, q_len, all_seq_len]
        # LLaVA 的输出 output.attentions 是一个元组，长度为生成的 token 数量。
        # 每一步的 attention 是一个元组，包含各层的 attention 张量。
        if not hasattr(outputs, "attentions"):
            raise RuntimeError("generate 输出中缺少 attentions。")

        # LLaVA 1.5 默认将图片编码为 576 个 patches (24x24)
        num_image_tokens = 576
        image_token_id = 32000  # LLaVA 中的 <image> token ID

        input_ids = inputs["input_ids"][0]
        image_idx = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        if len(image_idx) == 0:
            raise RuntimeError("未在 input_ids 中找到 <image> token。")

        image_start_idx = image_idx[0].item()
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
            img_attn = token_attn[image_start_idx : image_start_idx + num_image_tokens]
            all_image_attn.append(img_attn)

        # 将所有生成 token 对图片的 attention 叠加并取平均
        avg_image_attn = torch.stack(all_image_attn).mean(dim=0)
        # 转换成 24x24 的 2D Numpy 数组
        cross_attention_raw = (
            avg_image_attn.reshape(24, 24).cpu().to(torch.float32).numpy()
        )

        # 清理中间张量以释放显存
        del all_image_attn
        del avg_image_attn

        # 3. 提取 Vision Tower 的自注意力 (视觉模块最关注的地方)
        if "pixel_values" not in inputs:
            raise RuntimeError(
                "inputs 中缺少 pixel_values，无法提取 vision attention。"
            )

        with torch.no_grad():
            # 显式调用 vision tower 提取注意力
            vision_outputs = self.model.model.vision_tower(
                inputs["pixel_values"], output_attentions=True
            )

        # 取最后一层的 attention: shape (batch, heads, seq_len, seq_len)
        last_layer_vision_attn = vision_outputs.attentions[-1]
        # 在所有的注意力头上取平均: shape (seq_len, seq_len)
        avg_vision_attn = last_layer_vision_attn.mean(dim=1).squeeze(0)

        # CLIP ViT 的第一个 token 是 CLS token
        # 提取 CLS token 对图像 patches (索引 1 到最后) 的注意力
        cls_attn = avg_vision_attn[0, 1:]
        vision_attention_raw = cls_attn.reshape(24, 24).cpu().to(torch.float32).numpy()

        del last_layer_vision_attn
        del avg_vision_attn

        # 将两种注意力打包返回
        attention_dict = {
            "cross_attention_raw": cross_attention_raw,
            "vision_attention_raw": vision_attention_raw,
            # 兼容现有热力图流程，同时明确这些字段仅用于可视化。
            "cross_attention": normalize_attention_for_visualization(
                np.log(cross_attention_raw + 1e-6)
            ),
            "vision_attention": normalize_attention_for_visualization(
                np.log(vision_attention_raw + 1e-6)
            ),
        }

        return generated_text, attention_dict
