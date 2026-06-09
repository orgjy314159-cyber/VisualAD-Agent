"""
质检数据结构定义。

使用 Pydantic 定义统一的输入输出数据结构，
确保各工具模块之间的数据传递类型安全、可序列化。
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ImageQualityResult(BaseModel):
    """图像质量检测结果。"""

    brightness: float = Field(..., description="亮度均值（灰度图均值）")
    contrast: float = Field(..., description="对比度（灰度图标准差）")
    blur_score: float = Field(..., description="模糊程度（Laplacian 方差）")
    is_blurry: bool = Field(..., description="是否模糊")
    is_too_dark: bool = Field(..., description="是否过暗")
    is_too_bright: bool = Field(..., description="是否过曝")
    quality_level: str = Field(..., description="图像质量等级：good / warning / poor")
    message: str = Field(..., description="图像质量综合描述")


class DefectItem(BaseModel):
    """单个缺陷区域信息。"""

    bbox: List[int] = Field(..., description="缺陷边界框 [x, y, w, h]")
    area: float = Field(..., description="缺陷区域面积（像素）")
    confidence: float = Field(..., description="缺陷置信度，范围 [0, 1]")
    defect_type: str = Field(..., description="缺陷类型标签")


class AnomalyDetectionResult(BaseModel):
    """异常检测结果。"""

    has_defect: bool = Field(..., description="是否存在疑似缺陷")
    defect_count: int = Field(..., description="检测到的缺陷数量")
    defects: List[DefectItem] = Field(default_factory=list, description="缺陷列表")
    global_anomaly_score: float = Field(..., description="全局异常分数，范围 [0, 1]")


class VisualizationResult(BaseModel):
    """检测可视化结果。"""

    visualization_path: str = Field(..., description="可视化结果图保存路径")


class VLMExplanationResult(BaseModel):
    """多模态解释结果。"""

    explanation: str = Field(..., description="自然语言质检解释")


class InspectionReport(BaseModel):
    """最终质检报告。

    整合图像质量、异常检测、可视化和解释结果的完整报告。
    """

    image_path: str = Field(..., description="检测图像路径")
    user_query: str = Field(..., description="用户输入的检测任务描述")
    quality_result: ImageQualityResult = Field(..., description="图像质量检测结果")
    anomaly_result: AnomalyDetectionResult = Field(..., description="异常检测结果")
    visualization_result: Optional[VisualizationResult] = Field(
        default=None, description="可视化结果"
    )
    vlm_result: VLMExplanationResult = Field(..., description="多模态解释结果")
    final_decision: str = Field(
        ..., description="最终判定：Pass / Suspected Defect / Need Recheck"
    )
    severity: str = Field(..., description="严重程度：low / medium / high")
    markdown_report: str = Field(..., description="Markdown 格式的质检报告")


# ---------- __main__ 测试 ----------
if __name__ == "__main__":
    import json

    # 构造一个完整的 InspectionReport 示例
    quality = ImageQualityResult(
        brightness=126.5,
        contrast=42.1,
        blur_score=156.8,
        is_blurry=False,
        is_too_dark=False,
        is_too_bright=False,
        quality_level="good",
        message="图像质量良好，可以进行后续检测。",
    )

    defect = DefectItem(
        bbox=[120, 80, 220, 160],
        area=4200.0,
        confidence=0.72,
        defect_type="suspected_surface_anomaly",
    )

    anomaly = AnomalyDetectionResult(
        has_defect=True,
        defect_count=1,
        defects=[defect],
        global_anomaly_score=0.68,
    )

    visualization = VisualizationResult(
        visualization_path="outputs/visualizations/demo_vis.png",
    )

    vlm = VLMExplanationResult(
        explanation="系统在图像中检测到 1 个疑似异常区域，"
        "主要集中在图像中部，建议进入人工复核流程。",
    )

    report = InspectionReport(
        image_path="data/demo_images/example.jpg",
        user_query="请判断这张图像是否存在表面缺陷",
        quality_result=quality,
        anomaly_result=anomaly,
        visualization_result=visualization,
        vlm_result=vlm,
        final_decision="Suspected Defect",
        severity="medium",
        markdown_report="# 质检报告\n\n检测到 1 个疑似缺陷。",
    )

    # 打印 JSON
    print(report.model_dump_json(indent=2))
