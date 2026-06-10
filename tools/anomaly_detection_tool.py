"""
轻量异常检测工具。

支持两种检测方法：
- baseline: 基于 OpenCV 的背景残差 + 自适应阈值（默认）
- yolo: 基于 YOLOv8 的自定义训练模型
"""

import sys
import os
import cv2
import numpy as np
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from schemas.inspection_schema import AnomalyDetectionResult, DefectItem
from utils.image_io import load_image
from config import MIN_CONTOUR_AREA, GAUSSIAN_KERNEL, YOLO_MODEL_PATH

# baseline 内部参数
_RESIDUAL_THRESH_FACTOR = 2.5
_BACKGROUND_BLUR_SIZE = 31
_MORPH_KERNEL_SIZE = 3


def detect_anomalies(
    image_path: str,
    method: str = "baseline",
    model_path: str = None,
    yolo_conf: float = 0.25,
) -> AnomalyDetectionResult:
    """检测图像中的疑似异常区域。

    Args:
        image_path: 图像文件路径。
        method: 检测方法，可选 "baseline"（OpenCV 规则）或 "yolo"（YOLOv8 模型）。
        model_path: YOLO 模型 .pt 文件路径，仅 method="yolo" 时生效。
        yolo_conf: YOLO 置信度阈值，范围 [0, 1]，默认 0.25。

    Returns:
        AnomalyDetectionResult: 异常检测结果。
    """
    if method == "yolo":
        return _detect_yolo(image_path, model_path, yolo_conf)
    return _detect_baseline(image_path)


# ==================== baseline 方法 ====================

def _detect_baseline(image_path: str) -> AnomalyDetectionResult:
    """OpenCV 背景残差法：灰度图 → 背景建模 → 残差 → 阈值 → 轮廓。"""
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    background = cv2.GaussianBlur(denoised, (_BACKGROUND_BLUR_SIZE, _BACKGROUND_BLUR_SIZE), 0)
    residual = np.abs(denoised.astype(np.float32) - background.astype(np.float32))

    mean_residual = residual.mean()
    std_residual = residual.std()
    threshold = mean_residual + _RESIDUAL_THRESH_FACTOR * std_residual
    _, anomaly_mask = cv2.threshold(
        residual.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects: list = []
    image_area = gray.shape[0] * gray.shape[1]

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox = [int(x), int(y), int(w), int(h)]

        roi = gray[y : y + h, x : x + w]
        roi_bg = background[y : y + h, x : x + w]
        local_contrast = float(np.abs(roi.astype(np.float32) - roi_bg.astype(np.float32)).mean())

        contrast_score = min(local_contrast / 32.0, 1.0) * 0.6
        area_ratio = area / image_area
        area_score = min(area_ratio * 200, 1.0) * 0.4
        confidence = round(min(contrast_score + area_score, 1.0), 4)

        defects.append(
            DefectItem(
                bbox=bbox,
                area=round(area, 1),
                confidence=confidence,
                defect_type="suspected_surface_anomaly",
            )
        )

    return _build_result(defects, image_area)


# ==================== YOLO 方法 ====================

def _detect_yolo(
    image_path: str,
    model_path: str = None,
    conf: float = 0.25,
) -> AnomalyDetectionResult:
    """使用 YOLOv8 模型进行缺陷检测。

    Args:
        image_path: 图像文件路径。
        model_path: YOLO .pt 模型路径，None 则使用 yolov8n.pt。
        conf: 置信度阈值。

    Returns:
        AnomalyDetectionResult: 异常检测结果。
    """
    # 尝试导入 ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "YOLO 模式需要 ultralytics 库，但当前环境未安装。"
            "\n请确保 requirements.txt 中包含 torch 和 ultralytics。"
            "\n本地安装: pip install torch ultralytics"
        )

    # 确定模型路径：优先用传入的，其次用配置文件中的
    if model_path is None:
        model_path = YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO 模型文件不存在: {model_path}"
            "\n请将 .pt 模型文件放入 models/ 目录，或在侧边栏上传。"
        )

    # 加载模型并推理
    model = YOLO(model_path)
    results = model(image_path, conf=conf, verbose=False)

    # 读取图像以获取尺寸
    image = load_image(image_path)
    image_area = image.shape[0] * image.shape[1]

    defects: list = []
    class_names = model.names if hasattr(model, "names") else {}

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # xyxy → [x, y, w, h]
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = float((x2 - x1) * (y2 - y1))
            confidence = round(float(box.conf[0]), 4)
            cls_id = int(box.cls[0])
            defect_type = class_names.get(cls_id, f"class_{cls_id}")

            defects.append(
                DefectItem(
                    bbox=bbox,
                    area=round(area, 1),
                    confidence=confidence,
                    defect_type=defect_type,
                )
            )

    return _build_result(defects, image_area)


# ==================== 公共结果构建 ====================

def _build_result(defects: list, image_area: float) -> AnomalyDetectionResult:
    """根据缺陷列表构建 AnomalyDetectionResult。"""
    if defects:
        total_defect_area = sum(d.area for d in defects)
        avg_confidence = sum(d.confidence for d in defects) / len(defects)
        area_ratio = total_defect_area / image_area
        count_factor = min(len(defects) / 5.0, 1.0)
        score = area_ratio * 15 + count_factor * 0.25 + avg_confidence * 0.3
        global_anomaly_score = round(min(score, 1.0), 4)
        has_defect = True
    else:
        global_anomaly_score = 0.0
        has_defect = False

    return AnomalyDetectionResult(
        has_defect=has_defect,
        defect_count=len(defects),
        defects=defects,
        global_anomaly_score=global_anomaly_score,
    )


# ---------- __main__ 测试 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轻量异常检测工具")
    parser.add_argument("--image", type=str, required=True, help="图像文件路径")
    parser.add_argument(
        "--method",
        type=str,
        default="baseline",
        choices=["baseline", "yolo"],
        help="检测方法: baseline（OpenCV 规则）| yolo（YOLOv8 模型）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO .pt 模型路径（仅 method=yolo 时生效）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO 置信度阈值（默认 0.25）",
    )
    args = parser.parse_args()

    print(f"[INFO] 检测方法: {args.method}", file=sys.stderr)
    result = detect_anomalies(
        args.image,
        method=args.method,
        model_path=args.model,
        yolo_conf=args.conf,
    )
    print(result.model_dump_json(indent=2))
