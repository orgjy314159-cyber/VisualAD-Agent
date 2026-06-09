"""
可视化工具。

在原始图像上绘制检测结果：bbox、置信度标注、异常区域 mask 叠加。
支持无缺陷时的"No obvious defect"标注。
"""

import sys
import os
import cv2
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from schemas.inspection_schema import AnomalyDetectionResult, VisualizationResult
from utils.image_io import load_image, save_image
from utils.file_utils import make_output_path

# 可视化样式参数
_BBOX_COLOR = (0, 0, 255)         # 红色 bbox
_BBOX_THICKNESS = 2
_MASK_COLOR = (0, 0, 255)         # 红色 mask 叠加
_MASK_ALPHA = 0.3                 # mask 透明度
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_FONT_COLOR = (255, 255, 255)     # 白色文字
_LABEL_BG_COLOR = (0, 0, 255)     # 标签背景色


def visualize_detection(
    image_path: str,
    anomaly_result: AnomalyDetectionResult,
    output_dir: str = "outputs/visualizations",
) -> VisualizationResult:
    """在图像上绘制异常检测结果并保存。

    功能：
    1. 在原图上绘制缺陷 bbox
    2. 在 bbox 附近标注 confidence
    3. 叠加异常区域半透明 mask
    4. 如果没有缺陷，在左上角标注"No obvious defect"
    5. 保存结果到指定目录，文件名含时间戳

    Args:
        image_path: 原始图像路径。
        anomaly_result: 异常检测结果。
        output_dir: 输出目录路径。

    Returns:
        VisualizationResult: 可视化结果（含保存路径）。
    """
    # 读取图像
    image = load_image(image_path)
    vis_image = image.copy()

    if anomaly_result.has_defect and anomaly_result.defects:
        # 为每个缺陷绘制 bbox 和标签
        for i, defect in enumerate(anomaly_result.defects):
            x, y, w, h = defect.bbox

            # 绘制 bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), _BBOX_COLOR, _BBOX_THICKNESS)

            # 绘制置信度标签
            label = f"#{i + 1} {defect.confidence:.2f}"
            _draw_label(vis_image, label, x, y - 2)

        # 叠加异常区域半透明 mask
        mask = _create_defect_mask(image.shape, anomaly_result.defects)
        vis_image = _overlay_mask(vis_image, mask, _MASK_COLOR, _MASK_ALPHA)

        # 在左上角标注缺陷统计
        summary = f"Defects: {anomaly_result.defect_count} | Score: {anomaly_result.global_anomaly_score:.2f}"
    else:
        # 无缺陷标注
        summary = "No obvious defect"
        cv2.putText(
            vis_image,
            summary,
            (10, 30),
            _FONT,
            0.7,
            (0, 255, 0),  # 绿色
            2,
        )

    # 在图像顶部绘制概述
    if anomaly_result.has_defect:
        cv2.putText(
            vis_image,
            summary,
            (10, 25),
            _FONT,
            0.6,
            (0, 255, 255),  # 黄色
            2,
        )

    # 保存结果
    stem = os.path.splitext(os.path.basename(image_path))[0]
    save_path = make_output_path(output_dir, f"{stem}_vis", ".png")
    save_image(vis_image, save_path)

    return VisualizationResult(visualization_path=save_path)


def _draw_label(image, text: str, x: int, y: int):
    """在图像上绘制带背景的文本标签。

    Args:
        image: 目标图像（原地修改）。
        text: 标签文本。
        x: 文本左下角 x 坐标。
        y: 文本左下角 y 坐标。
    """
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)

    # 确保标签在图像内
    label_y = max(y, th + baseline)

    # 绘制背景矩形
    cv2.rectangle(
        image,
        (x, label_y - th - baseline),
        (x + tw, label_y + baseline),
        _LABEL_BG_COLOR,
        -1,
    )

    # 绘制文字
    cv2.putText(
        image,
        text,
        (x, label_y),
        _FONT,
        _FONT_SCALE,
        _FONT_COLOR,
        _FONT_THICKNESS,
    )


def _create_defect_mask(image_shape, defects) -> "np.ndarray":
    """根据缺陷列表生成二值 mask。

    Args:
        image_shape: 图像尺寸 (H, W, C)。
        defects: DefectItem 列表。

    Returns:
        np.ndarray: 二值 mask。
    """
    import numpy as np

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for defect in defects:
        x, y, w, h = defect.bbox
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def _overlay_mask(image, mask, color: tuple, alpha: float) -> "np.ndarray":
    """将彩色 mask 以半透明方式叠加到图像上。

    Args:
        image: 原始图像。
        mask: 二值 mask。
        color: 叠加颜色 (B, G, R)。
        alpha: 透明度，范围 [0, 1]。

    Returns:
        np.ndarray: 叠加后的图像。
    """
    import numpy as np

    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


# ---------- __main__ 测试 ----------
if __name__ == "__main__":
    from anomaly_detection_tool import detect_anomalies

    parser = argparse.ArgumentParser(description="可视化工具")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="图像文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="输出目录（默认: outputs/visualizations）",
    )
    args = parser.parse_args()

    # 先执行异常检测
    print(f"[INFO] 执行异常检测: {args.image}")
    anomaly_result = detect_anomalies(args.image)
    print(f"[INFO] 检测到 {anomaly_result.defect_count} 个疑似缺陷")

    # 生成可视化
    vis_result = visualize_detection(args.image, anomaly_result, args.output_dir)
    print(f"[INFO] 可视化已保存: {vis_result.visualization_path}")
    print(vis_result.model_dump_json(indent=2))
