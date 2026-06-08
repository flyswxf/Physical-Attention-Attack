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
import sys
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
    """检查 data 目录结构是否正确（仅校验需要上传的子目录）"""
    missing = []
    for d in dirs_to_include:
        if not os.path.isdir(os.path.join(DATA_DIR, d)):
            missing.append(d)
    if missing:
        print(f"[ERROR] data 目录缺少以下子目录: {missing}")
        print(f"       完整路径: {DATA_DIR}")
        sys.exit(1)

    # 统计文件数
    for d in dirs_to_include:
        full = os.path.join(DATA_DIR, d)
        count = sum(
            1 for _ in os.listdir(full) if os.path.isfile(os.path.join(full, _))
        )
        print(f"  {d}: {count} 个文件")


def upload(version_notes, owner, dataset, include_dirs=None):
    """上传数据集到 Kaggle（首次创建，后续自动发布新版本）"""
    if include_dirs is None:
        include_dirs = INCLUDE_DIRS

    print("[INFO] 检查 data 目录...")
    validate_data_dir(include_dirs)

    staging_dir = tempfile.mkdtemp(prefix="kaggle_staging_")
    print(f"[INFO] 暂存目录: {staging_dir}")
    try:
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
    except Exception as e:
        print(f"\n[ERROR] 上传失败: {e}")
        print("[HINT] 请确认：")
        print("  1. Kaggle API Token 已正确配置（~/.kaggle/kaggle.json）")
        print("  2. 你有该数据集的写入权限")
        sys.exit(1)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print("[INFO] 已清理暂存目录")


def main():
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
