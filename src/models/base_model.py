from abc import ABC, abstractmethod


class BaseModel(ABC):
    """多模态模型封装基类，约定统一的加载与推理接口。"""

    def __init__(self, model_id, device="cuda"):
        """
        功能描述:
        - 初始化模型基类的公共属性，包括模型标识、运行设备和延迟加载的实例句柄。

        输入参数:
        - model_id (str): 模型名称、仓库 ID 或本地路径。
        - device (str): 期望使用的运行设备，默认值为 `"cuda"`。

        返回值:
        - None: 构造函数仅初始化对象状态。
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """
        功能描述:
        - 加载模型主体及其配套处理器或分词器。

        输入参数:
        - 无。

        返回值:
        - bool: `True` 表示模型加载成功。
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference_and_get_attention(self, image, prompt):
        """
        功能描述:
        - 对输入图像和文本提示执行推理，并返回生成文本与注意力信息。

        输入参数:
        - image (PIL.Image.Image): 输入图像对象。
        - prompt (str): 提供给模型的文本提示。

        返回值:
        - tuple[str, dict]: 第一个元素为生成文本，第二个元素为注意力图数据字典。
        """
        raise NotImplementedError
