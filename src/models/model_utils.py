from .llava_model import LLaVAModel

# 模型注册表，后续如果添加其他模型，直接在这里注册
MODEL_REGISTRY = {
    "llava": LLaVAModel,
    # "qwen": QwenModel,
}


def get_model(model_family, model_id=None, device="cuda"):
    """
    功能描述:
    - 根据模型系列名称创建对应的模型实例，并在需要时覆盖默认模型 ID。

    输入参数:
    - model_family (str): 模型系列名称，例如 `"llava"`。
    - model_id (str | None): 具体的模型 ID 或本地路径；为 `None` 时使用默认值。
    - device (str): 模型运行设备，默认值为 `"cuda"`。

    返回值:
    - BaseModel: 已实例化但尚未加载权重的模型对象。
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
