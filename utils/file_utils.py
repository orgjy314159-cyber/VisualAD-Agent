"""
文件工具。

提供目录创建、时间戳生成和输出路径构建等辅助功能。
"""

import os
from datetime import datetime


def ensure_dir(path: str):
    """确保目录存在，不存在则自动递归创建。

    Args:
        path: 目录路径。
    """
    if path:
        os.makedirs(path, exist_ok=True)


def get_timestamp_string() -> str:
    """返回当前时间戳字符串，用于文件名。

    Returns:
        str: 格式为 YYYYMMDD_HHMMSS 的时间戳。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_path(output_dir: str, stem: str, suffix: str) -> str:
    """构建带时间戳的输出文件路径。

    自动确保输出目录存在。

    Args:
        output_dir: 输出目录路径。
        stem: 文件名主体（不含扩展名）。
        suffix: 文件扩展名（含点，如 ".png" 或 ".json"）。

    Returns:
        str: 完整输出路径，格式为 output_dir/stem_timestamp.suffix。
    """
    ensure_dir(output_dir)
    timestamp = get_timestamp_string()
    filename = f"{stem}_{timestamp}{suffix}"
    return os.path.join(output_dir, filename)
