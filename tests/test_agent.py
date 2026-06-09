"""
Agent 主调度器测试。

测试 VisualInspectionAgent.run 端到端流程的稳定性和输出正确性。
"""

import sys
import os
import pytest
import cv2
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agent import VisualInspectionAgent
from schemas.inspection_schema import InspectionReport


def _save_temp(img: np.ndarray, tmp_path, name: str = "test.jpg") -> str:
    """将图像保存到临时目录。"""
    path = os.path.join(str(tmp_path), name)
    cv2.imwrite(path, img)
    return path


class TestVisualInspectionAgent:
    """VisualInspectionAgent 端到端测试。"""

    def test_run_returns_inspection_report(self, tmp_path):
        """run 应返回 InspectionReport 实例。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        path = _save_temp(img, tmp_path)

        agent = VisualInspectionAgent()
        report = agent.run(path, "测试检测")

        assert isinstance(report, InspectionReport)

    def test_report_has_all_required_fields(self, tmp_path):
        """返回的报告应包含所有必要字段。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        img[60:80, 80:160] = 40  # 添加异常
        path = _save_temp(img, tmp_path)

        agent = VisualInspectionAgent()
        report = agent.run(path, "检查缺陷")

        assert report.image_path == path
        assert report.user_query == "检查缺陷"
        assert report.final_decision in ("Pass", "Suspected Defect", "Need Recheck")
        assert report.severity in ("low", "medium", "high")
        assert report.quality_result is not None
        assert report.anomaly_result is not None
        assert report.vlm_result is not None
        assert isinstance(report.markdown_report, str)
        assert len(report.markdown_report) > 0

    def test_markdown_report_has_expected_sections(self, tmp_path):
        """Markdown 报告应包含所有必要章节。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        path = _save_temp(img, tmp_path)

        agent = VisualInspectionAgent()
        report = agent.run(path, "测试")

        assert "检测结论" in report.markdown_report
        assert "图像质量" in report.markdown_report
        assert "缺陷检测" in report.markdown_report
        assert "人工复核" in report.markdown_report

    def test_run_with_default_query(self, tmp_path):
        """不传 user_query 应使用默认值。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        path = _save_temp(img, tmp_path)

        agent = VisualInspectionAgent()
        report = agent.run(path, "")

        assert isinstance(report, InspectionReport)
        assert len(report.user_query) > 0  # 应该被填了默认值

    def test_nonexistent_file_raises(self):
        """不存在的文件应抛出 FileNotFoundError。"""
        agent = VisualInspectionAgent()
        with pytest.raises(FileNotFoundError):
            agent.run("nonexistent_file_12345.jpg")

    def test_visualization_result_exists(self, tmp_path):
        """可视化结果路径应存在。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        img[80:100, 100:150] = 30
        path = _save_temp(img, tmp_path)

        agent = VisualInspectionAgent()
        report = agent.run(path, "测试")

        if report.visualization_result:
            assert os.path.exists(report.visualization_result.visualization_path)

    def test_json_serializable(self, tmp_path):
        """报告应可序列化为 JSON。"""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 150
        path = _save_temp(img, tmp_path)

        agent = VisualInspectionAgent()
        report = agent.run(path, "测试")

        json_str = report.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
