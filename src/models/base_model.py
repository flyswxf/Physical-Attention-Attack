from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_id, device="cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """
        加载模型和相关处理器/分词器。
        返回布尔值表示是否加载成功。
        """
        pass

    @abstractmethod
    def run_inference_and_get_attention(self, image, prompt):
        """
        运行推理并提取注意力权重。
        
        Args:
            image: 输入的图像
            prompt: 文本提示
            
        Returns:
            generated_text: 生成的文本
            attention_maps: 注意力图数据
        """
        pass
