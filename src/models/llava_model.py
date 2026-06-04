import torch
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

        print(f"Loading {self.model_id} onto {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # 为了提取attention，保持默认或者用float16。
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(self.device)
        
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
        
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.no_grad():
            # 设置 output_attentions=True 获取注意力图
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # 1. 解码生成的文本
        generated_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # 2. 提取 Attention Maps
        # 注意: LLaVA内部有视觉编码器(ViT)和语言模型(LLM)。
        # outputs.attentions 通常是LLM层面的注意力（包括对图像Token的交叉注意力）
        # 如果要提取纯ViT的注意力，可能需要注册hook。
        # 这里我们返回所有的注意力数据供下游 vis_utils 分析
        attention_maps = outputs.attentions if hasattr(outputs, 'attentions') else None
        
        return generated_text, attention_maps
