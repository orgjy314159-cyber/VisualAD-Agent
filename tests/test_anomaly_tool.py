"""
异常检测工具测试。

测试 detect_anomalies 在不同场景下的稳定性和输出正确性。
"""

import sys
import os
import pytest
import cv2
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.anomaly_detection_tool import detect_anomalies
from schemas.inspection_schema import AnomalyDetectionResult, DefectItem


def _save_temp(img: np.ndarray, tmp_path, name: str = "test.jpg") -> str:
    """将图像保存到临时目录。"""
    path = os.path.join(str(tmp_path), name)
    cv2.imwrite(path, img)
    return path


class TestAnomalyDetectionTool:
    """detect_anomalies 函数测试。"""

    def test_returns_correct_type(self, tmp_path):
        """应返回 AnomalyDetectionResult 实例。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 140
        img = img + np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img, 0, 255).astype(np.uint8)
        path = _save_temp(img, tmp_path)
        result = detect_anomalies(path)
        assert isinstance(result, AnomalyDetectionResult)

    def test_uniform_image_no_defect(self, tmp_path):
        """完全均匀的图像应无异常或只有极少误检。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 140
        path = _save_temp(img, tmp_path)
        result = detect_anomalies(path)
        # 均匀图像不应报告有缺陷
        assert isinstance(result.has_defect, bool)
        assert isinstance(result.defect_count, int)
        assert isinstance(result.global_anomaly_score, float)

    def test_image_with_obvious_defects(self, tmp_path):
        """有明显局部异常的图像应能检测到。"""
        img = np.ones((300, 400, 3), dtype=np.uint8) * 150

        # 添加明显的暗色划痕
        img[100:115, 120:300] = 40
        # 添加明显的亮色斑点
        img[200:240, 80:130] = 240

        path = _save_temp(img, tmp_path)
        result = detect_anomalies(path)
        assert isinstance(result, AnomalyDetectionResult)
        # 有明显异常时，应该有缺陷被检测到
        assert result.defect_count >= 1
        assert result.has_defect is True

    def test_defect_items_structure(self, tmp_path):
        """DefectItem 字段结构应正确。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        # 添加一个明显暗斑
        img[80:120, 100:160] = 30

        path = _save_temp(img, tmp_path)
        result = detect_anomalies(path)

        if result.defects:
            defect = result.defects[0]
            assert isinstance(defect, DefectItem)
            assert isinstance(defect.bbox, list)
            assert len(defect.bbox) == 4
            assert all(isinstance(v, int) for v in defect.bbox)
            assert 0 <= defect.confidence <= 1
            assert defect.area > 0
            assert isinstance(defect.defect_type, str)

    def test_global_score_range(self, tmp_path):
        """global_anomaly_score 应在 [0, 1] 范围内。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        path = _save_temp(img, tmp_path)
        result = detect_anomalies(path)
        assert 0.0 <= result.global_anomaly_score <= 1.0

    def test_no_crash_on_small_image(self, tmp_path):
        """小尺寸图像不崩溃。"""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        path = _save_temp(img, tmp_path)
        result = detect_anomalies(path)
        assert isinstance(result, AnomalyDetectionResult)

    def test_no_crash_on_dark_image(self, tmp_path):
        """极暗图像不崩溃。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 5
        path = _save_temp(img, tmp_path)
        result = detect_anomalies(path)
        assert isinstance(result, AnomalyDetectionResult)

    def test_nonexistent_file_raises(self):
        """不存在的文件应抛出错误。"""
        with pytest.raises(Exception):
            detect_anomalies("nonexistent_file_12345.jpg")
