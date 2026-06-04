"""
Kaggle 数据集上传/更新工具

用法:
    # 首次上传
    python src/utils/kaggle_upload.py --upload

    # 更新数据集（创建新版本）
    python src/utils/kaggle_upload.py --update --notes "新增了一批攻击图片"

    # 指定自定义 owner/dataset 名称
    python src/utils/kaggle_upload.py --upload --owner myusername --dataset my-dataset

前置条件:
    pip install kagglehub
    # 还需配置 Kaggle API Key:
    #   - 方式1: 将 kaggle.json 放在 ~/.kaggle/ 目录下
    #   - 方式2: 设置环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ===================== 配置区域（按需修改） =====================
DEFAULT_OWNER = "wangyufei77"  # 替换为你的 Kaggle 用户名
DEFAULT_DATASET = "physical-attention-attack"  # 数据集名称（Kaggle 上显示的名称）
DATASET_TITLE = "Physical Attention Attack Dataset"
DATASET_DESCRIPTION = """
Dataset for the Physical Attention Attack experiment on LLM safety.

Contains three subdirectories:
- clean_images: Original images before text injection
- attack_images: Images with injected attack text
- results: Experiment outputs including attention heatmaps, success/fail classifications
""".strip()
LICENSE_NAME = "MIT"
# =================================================================


def create_metadata_json(output_path, owner, dataset):
    """生成 dataset-metadata.json"""
    metadata = {
        "title": DATASET_TITLE,
        "id": f"{owner}/{dataset}",
        "licenses": [{"name": LICENSE_NAME}],
        "subtitle": DATASET_DESCRIPTION,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 已生成 metadata 文件: {output_path}")


def validate_data_dir():
    """检查 data 目录结构是否正确"""
    required_dirs = ["attack_images", "clean_images", "results", "fonts"]
    missing = []
    for d in required_dirs:
        if not os.path.isdir(os.path.join(DATA_DIR, d)):
            missing.append(d)
    if missing:
        print(f"[ERROR] data 目录缺少以下子目录: {missing}")
        print(f"       完整路径: {DATA_DIR}")
        sys.exit(1)

    # 统计文件数
    for d in required_dirs:
        full = os.path.join(DATA_DIR, d)
        count = sum(
            1 for _ in os.listdir(full) if os.path.isfile(os.path.join(full, _))
        )
        print(f"  {d}: {count} 个文件")


def upload_or_update(version_notes, owner, dataset, is_update=False):
    """
    上传或更新 Kaggle 数据集

    参数:
        version_notes: 版本说明
        owner: Kaggle 用户名
        dataset: 数据集名称
        is_update: True 表示更新已有数据集，False 表示首次上传
    """
    try:
        import kagglehub
    except ImportError:
        print("[ERROR] 未安装 kagglehub，请执行: pip install kagglehub")
        sys.exit(1)

    # 1. 校验数据目录
    print("[INFO] 检查 data 目录...")
    validate_data_dir()

    # 2. 生成 metadata 文件
    metadata_path = os.path.join(DATA_DIR, "dataset-metadata.json")
    create_metadata_json(metadata_path, owner, dataset)

    # 3. 上传/更新
    handle = f"{owner}/{dataset}"
    action = "更新" if is_update else "上传"

    print(f"\n[INFO] 准备{action}数据集到：{handle}")
    print(f"[INFO] 数据目录: {DATA_DIR}")
    print(f"[INFO] 版本说明: {version_notes}")

    try:
        kagglehub.dataset_upload(
            handle=handle,
            local_dataset_dir=DATA_DIR,
            version_notes=version_notes,
        )
        print(f"\n[SUCCESS] 数据集{action}成功！")
        print(f"  Kaggle 链接: https://www.kaggle.com/datasets/{handle}")
    except Exception as e:
        print(f"\n[ERROR] {action}失败: {e}")
        print("[HINT] 请确认：")
        print("  1. Kaggle API Token 已正确配置（~/.kaggle/access_token）")
        print("  2. 数据集名称尚未被占用（首次上传时）")
        print("  3. 你有该数据集的写入权限（更新时）")
        sys.exit(1)
    finally:
        # 清理临时 metadata 文件
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"[INFO] 已清理临时 metadata 文件")


def main():
    parser = argparse.ArgumentParser(description="Kaggle 数据集上传/更新工具")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="首次上传数据集到 Kaggle",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="更新已有的 Kaggle 数据集（创建新版本）",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="Automated dataset update",
        help="版本说明 (default: 'Automated dataset update')",
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

    if not args.upload and not args.update:
        parser.print_help()
        print("\n[ERROR] 请指定 --upload 或 --update")
        sys.exit(1)

    if args.upload and args.update:
        print("[ERROR] --upload 和 --update 不能同时使用")
        sys.exit(1)

    upload_or_update(
        version_notes=args.notes,
        owner=args.owner,
        dataset=args.dataset,
        is_update=args.update,
    )


if __name__ == "__main__":
    main()
