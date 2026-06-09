"""
图像读写工具。

提供稳健的图像加载与保存功能，自动处理路径和格式转换。
使用 imdecode/imencode 替代 imread/imwrite，支持含中文的路径。
"""

import os
import cv2
import numpy as np


def load_image(image_path: str):
    """加载图像，返回 OpenCV BGR 格式图像。

    使用 np.fromfile + cv2.imdecode 读取，支持含中文的路径。

    Args:
        image_path: 图像文件路径。

    Returns:
        numpy.ndarray: BGR 格式图像。

    Raises:
        FileNotFoundError: 当路径不存在时。
        ValueError: 当文件无法作为图像读取时。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 使用 imdecode 替代 imread，解决 Windows 下中文路径问题
    buf = np.fromfile(image_path, dtype=np.uint8)
    if len(buf) == 0:
        raise ValueError(f"图像文件为空: {image_path}")

    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"无法读取图像文件（可能格式不支持或文件损坏）: {image_path}"
        )

    return image


def load_image_rgb(image_path: str):
    """加载图像，返回 RGB 格式图像。

    Args:
        image_path: 图像文件路径。

    Returns:
        numpy.ndarray: RGB 格式图像。
    """
    bgr = load_image(image_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(image, save_path: str):
    """保存图像到指定路径，自动创建父目录。

    使用 cv2.imencode + tofile 写入，支持含中文的路径。

    Args:
        image: 要保存的图像（numpy.ndarray）。
        save_path: 保存路径。

    Raises:
        ValueError: 当图像为空时。
    """
    if image is None:
        raise ValueError("无法保存空图像")

    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # 根据扩展名确定编码格式
    ext = os.path.splitext(save_path)[1]
    if not ext:
        ext = ".png"

    # 使用 imencode + tofile 替代 imwrite，解决 Windows 下中文路径问题
    success, buf = cv2.imencode(ext, image)
    if not success:
        raise IOError(f"图像编码失败: {save_path}")

    buf.tofile(save_path)
