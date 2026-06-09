"""
图像质量检测工具。

使用 OpenCV 检测图像的基本质量指标：
亮度、对比度、模糊程度，并综合判断图像质量等级。
"""

import sys
import os
import cv2
import argparse

# 确保项目根目录在 sys.path 中（支持直接运行此文件）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from schemas.inspection_schema import ImageQualityResult
from utils.image_io import load_image
from config import BLUR_THRESHOLD, DARK_THRESHOLD, BRIGHT_THRESHOLD


def check_image_quality(image_path: str) -> ImageQualityResult:
    """检查图像质量，返回 ImageQualityResult。

    检测流程：
    1. 读取图像并转为灰度图
    2. 计算亮度均值（brightness）
    3. 计算对比度（灰度图标准差）
    4. 计算 Laplacian 方差作为模糊程度（blur_score）
    5. 根据阈值判定质量等级

    Args:
        image_path: 图像文件路径。

    Returns:
        ImageQualityResult: 图像质量检测结果。
    """
    # 读取图像并转灰度
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 亮度：灰度图均值
    brightness = float(gray.mean())

    # 对比度：灰度图标准差
    contrast = float(gray.std())

    # 模糊程度：Laplacian 方差，值越大越清晰
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 阈值判定
    is_blurry = blur_score < BLUR_THRESHOLD
    is_too_dark = brightness < DARK_THRESHOLD
    is_too_bright = brightness > BRIGHT_THRESHOLD

    # 质量等级
    if is_too_dark or is_too_bright or is_blurry:
        quality_level = "poor"
        message_parts = []
        if is_too_dark:
            message_parts.append("亮度过低")
        if is_too_bright:
            message_parts.append("亮度过曝")
        if is_blurry:
            message_parts.append("图像模糊")
        message = "，".join(message_parts) + "，建议重新采集图像。"
    elif blur_score < BLUR_THRESHOLD * 1.5:
        # 模糊分数偏低但未达到阈值，给 warning
        quality_level = "warning"
        message = "图像质量一般，可能存在轻微模糊，检测结果可能受影响。"
    elif brightness < DARK_THRESHOLD * 1.3 or brightness > BRIGHT_THRESHOLD * 0.9:
        quality_level = "warning"
        message = "图像亮度偏差较大，建议检查光照条件。"
    else:
        quality_level = "good"
        message = "图像质量良好，可以进行后续检测。"

    return ImageQualityResult(
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        blur_score=round(blur_score, 2),
        is_blurry=is_blurry,
        is_too_dark=is_too_dark,
        is_too_bright=is_too_bright,
        quality_level=quality_level,
        message=message,
    )


# ---------- __main__ 测试 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像质量检测工具")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="图像文件路径",
    )
    args = parser.parse_args()

    result = check_image_quality(args.image)
    print(result.model_dump_json(indent=2))
