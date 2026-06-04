from .llava_model import LLaVAModel

# 模型注册表，后续如果添加其他模型，直接在这里注册
MODEL_REGISTRY = {
    "llava": LLaVAModel,
    # "qwen": QwenModel,
}


def get_model(model_family, model_id=None, device="cuda"):
    """
    根据模型系列(model_family)获取实例化的模型对象。

    Args:
        model_family (str): 模型系列名称，例如 "llava"
        model_id (str, optional): 具体的模型ID/路径。如果为None，则使用模型的默认ID。
        device (str): 运行设备，默认为 "cuda"

    Returns:
        BaseModel: 实例化后的模型对象
    """
    model_family = model_family.lower()
    if model_family not in MODEL_REGISTRY:
        raise ValueError(
            f"不支持的模型系列: '{model_family}'。当前支持的模型系列: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_family]

    if model_id is not None:
        return model_class(model_id=model_id, device=device)
    return model_class(device=device)
