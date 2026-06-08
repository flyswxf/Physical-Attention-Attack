"""
Kaggle 数据集上传工具

用法:
    python src/utils/kaggle_upload.py
    python src/utils/kaggle_upload.py --notes "新增了一批图片"
    python src/utils/kaggle_upload.py --owner myusername --dataset my-dataset

前置条件:
    pip install kagglehub
    # 还需配置 Kaggle API Key:
    #   - 方式1: 将 kaggle.json 放在 ~/.kaggle/ 目录下
    #   - 方式2: 设置环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY
"""

import argparse
import json
import os
import shutil
import tempfile
import kagglehub

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ===================== 配置区域（按需修改） =====================
DEFAULT_OWNER = "wangyufei77"  # 替换为你的 Kaggle 用户名
DEFAULT_DATASET = "physical-attention-attack"  # 数据集名称（Kaggle 上显示的名称）
INCLUDE_DIRS = ["clean_images", "fonts"]  # 只上传这些子目录
DATASET_TITLE = "Physical Attention Attack Dataset"
DATASET_DESCRIPTION = """
Dataset for the Physical Attention Attack experiment on LLM safety.

Contains:
- clean_images: Original images before text injection
- fonts: Font files used for attack text injection
""".strip()
LICENSE_NAME = "MIT"
# =================================================================


def create_metadata_json(output_path, owner, dataset):
    """
    功能描述:
    - 为 Kaggle 数据集上传生成 `dataset-metadata.json` 元数据文件。

    输入参数:
    - output_path (str): 元数据文件输出路径。
    - owner (str): Kaggle 用户名。
    - dataset (str): Kaggle 数据集名称。

    返回值:
    - None: 函数直接将元数据写入磁盘。
    """
    metadata = {
        "title": DATASET_TITLE,
        "id": f"{owner}/{dataset}",
        "licenses": [{"name": LICENSE_NAME}],
        "subtitle": DATASET_DESCRIPTION,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 已生成 metadata 文件: {output_path}")


def validate_data_dir(dirs_to_include):
    """
    功能描述:
    - 校验上传所需的数据子目录是否存在，并打印每个目录中的文件数量。

    输入参数:
    - dirs_to_include (list[str]): 需要打包上传的 `data` 子目录名称列表。

    返回值:
    - None: 校验通过后仅打印目录统计信息。
    """
    missing = []
    for d in dirs_to_include:
        if not os.path.isdir(os.path.join(DATA_DIR, d)):
            missing.append(d)
    if missing:
        raise FileNotFoundError(
            f"data 目录缺少以下子目录: {missing}，完整路径: {DATA_DIR}"
        )

    # 统计文件数
    for d in dirs_to_include:
        full = os.path.join(DATA_DIR, d)
        count = sum(
            1 for _ in os.listdir(full) if os.path.isfile(os.path.join(full, _))
        )
        print(f"  {d}: {count} 个文件")


def upload(version_notes, owner, dataset, include_dirs=None):
    """
    功能描述:
    - 将指定的数据目录复制到暂存区并上传到 Kaggle 数据集。

    输入参数:
    - version_notes (str): 本次上传的版本说明。
    - owner (str): Kaggle 用户名。
    - dataset (str): Kaggle 数据集名称。
    - include_dirs (list[str] | None): 需要上传的 `data` 子目录列表；为 `None` 时使用默认配置。

    返回值:
    - None: 函数完成后直接打印 Kaggle 数据集链接。
    """
    if include_dirs is None:
        include_dirs = INCLUDE_DIRS

    print("[INFO] 检查 data 目录...")
    validate_data_dir(include_dirs)

    with tempfile.TemporaryDirectory(prefix="kaggle_staging_") as staging_dir:
        print(f"[INFO] 暂存目录: {staging_dir}")
        for d in include_dirs:
            src = os.path.join(DATA_DIR, d)
            dst = os.path.join(staging_dir, d)
            shutil.copytree(src, dst)
            print(f"  已复制: {d}")

        metadata_path = os.path.join(staging_dir, "dataset-metadata.json")
        create_metadata_json(metadata_path, owner, dataset)

        handle = f"{owner}/{dataset}"
        print(f"\n[INFO] 上传至: {handle}")
        print(f"[INFO] 版本说明: {version_notes}")

        kagglehub.dataset_upload(
            handle=handle,
            local_dataset_dir=staging_dir,
            version_notes=version_notes,
        )
        print(f"\n[SUCCESS] 上传成功！")
        print(f"  Kaggle 链接: https://www.kaggle.com/datasets/{handle}")
        print("[INFO] 已清理暂存目录")


def main():
    """
    功能描述:
    - 解析命令行参数并执行 Kaggle 数据集上传流程。

    输入参数:
    - 无。

    返回值:
    - None: 函数执行完成后直接在终端打印上传结果。
    """
    parser = argparse.ArgumentParser(description="Kaggle 数据集上传工具")
    parser.add_argument(
        "--notes", type=str, default="Automated dataset update", help="版本说明"
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=DEFAULT_OWNER,
        help=f"Kaggle 用户名 (default: {DEFAULT_OWNER})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"数据集名称 (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()

    upload(version_notes=args.notes, owner=args.owner, dataset=args.dataset)


if __name__ == "__main__":
    main()
