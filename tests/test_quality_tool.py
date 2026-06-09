"""
图像质量检测工具测试。

测试 check_image_quality 在各种图像条件下的稳定性和输出正确性。
"""

import sys
import os
import pytest
import cv2
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.image_quality_tool import check_image_quality
from schemas.inspection_schema import ImageQualityResult


def _make_image(brightness: int = 128, shape=(200, 300)) -> np.ndarray:
    """创建指定亮度的 BGR 测试图像。"""
    img = np.ones((*shape, 3), dtype=np.uint8) * brightness
    return img


def _save_temp(img: np.ndarray, tmp_path, name: str = "test.jpg") -> str:
    """将图像保存到临时目录。"""
    path = os.path.join(str(tmp_path), name)
    cv2.imwrite(path, img)
    return path


class TestImageQualityTool:
    """check_image_quality 函数测试。"""

    def test_returns_correct_type(self, tmp_path):
        """应返回 ImageQualityResult 实例。"""
        img = _make_image(128)
        path = _save_temp(img, tmp_path)
        result = check_image_quality(path)
        assert isinstance(result, ImageQualityResult)

    def test_normal_image_good_quality(self, tmp_path):
        """正常亮度的带纹理图像应判定为 good。"""
        # 纯色图无纹理会被判模糊，需加噪声模拟真实图像
        img = _make_image(140)
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        path = _save_temp(img, tmp_path)
        result = check_image_quality(path)
        assert not result.is_too_dark
        assert not result.is_too_bright
        # 带噪声的图像不应被判为 poor（除非噪声太小导致模糊判定）
        assert result.quality_level in ("good", "warning")

    def test_dark_image_detected(self, tmp_path):
        """暗图应被判定为过暗。"""
        img = _make_image(20)
        path = _save_temp(img, tmp_path)
        result = check_image_quality(path)
        assert result.is_too_dark
        assert result.brightness < 40

    def test_bright_image_detected(self, tmp_path):
        """亮图应被判定为过曝。"""
        img = _make_image(235)
        path = _save_temp(img, tmp_path)
        result = check_image_quality(path)
        assert result.is_too_bright
        assert result.brightness > 220

    def test_blurry_image_detected(self, tmp_path):
        """强模糊图像应被判定为模糊。"""
        img = _make_image(128)
        # 强高斯模糊
        img = cv2.GaussianBlur(img, (51, 51), 0)
        path = _save_temp(img, tmp_path)
        result = check_image_quality(path)
        assert result.is_blurry
        assert result.blur_score < 80

    def test_fields_are_valid(self, tmp_path):
        """所有字段应为有效值。"""
        img = _make_image(150)
        path = _save_temp(img, tmp_path)
        result = check_image_quality(path)
        assert isinstance(result.brightness, float)
        assert isinstance(result.contrast, float)
        assert isinstance(result.blur_score, float)
        assert result.blur_score >= 0
        assert result.contrast >= 0
        assert result.quality_level in ("good", "warning", "poor")
        assert isinstance(result.message, str)
        assert len(result.message) > 0

    def test_nonexistent_file_raises(self):
        """不存在的文件应抛出错误。"""
        with pytest.raises(Exception):
            check_image_quality("nonexistent_file_12345.jpg")

    def test_textured_image(self, tmp_path):
        """带纹理的图像应能正常处理。"""
        img = np.random.randint(60, 200, (200, 300, 3), dtype=np.uint8)
        path = _save_temp(img, tmp_path)
        result = check_image_quality(path)
        assert isinstance(result, ImageQualityResult)
        # 纹理图模糊度应较高
        assert result.blur_score > 0
