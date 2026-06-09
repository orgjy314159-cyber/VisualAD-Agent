"""
报告生成工具。

整合图像质量、异常检测、可视化和解释结果，
生成最终质检报告（JSON + Markdown 双格式）。
"""

import sys
import os
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from schemas.inspection_schema import (
    ImageQualityResult,
    AnomalyDetectionResult,
    VisualizationResult,
    VLMExplanationResult,
    InspectionReport,
)


def generate_inspection_report(
    image_path: str,
    user_query: str,
    quality_result: ImageQualityResult,
    anomaly_result: AnomalyDetectionResult,
    visualization_result: VisualizationResult,
    vlm_result: VLMExplanationResult,
) -> InspectionReport:
    """生成最终质检报告。

    包含 final_decision 判定、severity 分级和 Markdown 文本报告。

    Args:
        image_path: 检测图像路径。
        user_query: 用户检测任务描述。
        quality_result: 图像质量检测结果。
        anomaly_result: 异常检测结果。
        visualization_result: 可视化结果。
        vlm_result: 多模态解释结果。

    Returns:
        InspectionReport: 完整质检报告。
    """
    # 判定 final_decision
    final_decision = _determine_decision(quality_result, anomaly_result)

    # 判定 severity
    severity = _determine_severity(anomaly_result)

    # 生成 Markdown 报告
    markdown_report = _build_markdown_report(
        image_path=image_path,
        user_query=user_query,
        quality_result=quality_result,
        anomaly_result=anomaly_result,
        visualization_result=visualization_result,
        vlm_result=vlm_result,
        final_decision=final_decision,
        severity=severity,
    )

    return InspectionReport(
        image_path=image_path,
        user_query=user_query,
        quality_result=quality_result,
        anomaly_result=anomaly_result,
        visualization_result=visualization_result,
        vlm_result=vlm_result,
        final_decision=final_decision,
        severity=severity,
        markdown_report=markdown_report,
    )


def _determine_decision(
    quality_result: ImageQualityResult,
    anomaly_result: AnomalyDetectionResult,
) -> str:
    """根据质量与检测结果判定最终结论。

    Returns:
        str: "Pass" | "Suspected Defect" | "Need Recheck"
    """
    if quality_result.quality_level == "poor":
        return "Need Recheck"
    if anomaly_result.has_defect:
        return "Suspected Defect"
    return "Pass"


def _determine_severity(anomaly_result: AnomalyDetectionResult) -> str:
    """根据缺陷数量和分数判定严重程度。

    Returns:
        str: "low" | "medium" | "high"
    """
    if not anomaly_result.has_defect or anomaly_result.defect_count == 0:
        return "low"
    if anomaly_result.defect_count >= 2 or anomaly_result.global_anomaly_score >= 0.5:
        return "high"
    return "medium"


def _build_markdown_report(
    image_path: str,
    user_query: str,
    quality_result: ImageQualityResult,
    anomaly_result: AnomalyDetectionResult,
    visualization_result: VisualizationResult,
    vlm_result: VLMExplanationResult,
    final_decision: str,
    severity: str,
) -> str:
    """构建 Markdown 格式的质检报告。"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_name = os.path.basename(image_path)

    # 决策与严重程度的中文映射
    decision_cn = {
        "Pass": "通过",
        "Suspected Defect": "疑似缺陷",
        "Need Recheck": "需要重新检测",
    }
    severity_cn = {"low": "低", "medium": "中", "high": "高"}

    # 严重程度图标
    severity_icon = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}

    lines = [
        f"# 工业视觉质检报告",
        "",
        f"**生成时间**：{now}",
        f"**检测图像**：{image_name}",
        f"**用户任务**：{user_query}",
        "",
        "---",
        "",
        "## 检测结论",
        "",
        f"| 项目 | 结果 |",
        f"|------|------|",
        f"| 最终判定 | **{decision_cn.get(final_decision, final_decision)}** |",
        f"| 严重程度 | {severity_icon.get(severity, '')} **{severity_cn.get(severity, severity)}** |",
        f"| 全局异常分数 | {anomaly_result.global_anomaly_score:.4f} |",
        f"| 疑似缺陷数量 | {anomaly_result.defect_count} |",
        "",
        "---",
        "",
        "## 图像质量",
        "",
        f"| 指标 | 数值 | 状态 |",
        f"|------|------|------|",
        f"| 亮度均值 | {quality_result.brightness:.2f} | {'[!]过暗' if quality_result.is_too_dark else '[!]过曝' if quality_result.is_too_bright else '[OK] 正常'} |",
        f"| 对比度 | {quality_result.contrast:.2f} | — |",
        f"| 清晰度得分 | {quality_result.blur_score:.2f} | {'[!]模糊' if quality_result.is_blurry else '[OK] 清晰'} |",
        f"| 质量等级 | **{quality_result.quality_level}** | — |",
        "",
        f"> {quality_result.message}",
        "",
        "---",
        "",
        "## 缺陷检测详情",
        "",
    ]

    if anomaly_result.has_defect and anomaly_result.defects:
        lines.append(f"共检测到 **{anomaly_result.defect_count}** 个疑似异常区域：")
        lines.append("")
        lines.append(
            "| 编号 | 边界框 (x, y, w, h) | 面积 (px) | 置信度 | 类型 |"
        )
        lines.append(
            "|------|---------------------|-----------|--------|------|"
        )
        for i, defect in enumerate(anomaly_result.defects):
            x, y, w, h = defect.bbox
            lines.append(
                f"| #{i + 1} | [{x}, {y}, {w}, {h}] | {defect.area:.0f} | {defect.confidence:.2f} | {defect.defect_type} |"
            )
        lines.append("")
    else:
        lines.append("未检测到明显异常区域。")
        lines.append("")

    # 可视化引用
    if visualization_result:
        vis_name = os.path.basename(visualization_result.visualization_path)
        lines.append(f"**可视化结果**：`{vis_name}`")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 模型解释",
            "",
            vlm_result.explanation,
            "",
            "---",
            "",
            "## 人工复核建议",
            "",
        ]
    )

    # 根据场景给出不同复核建议
    if final_decision == "Need Recheck":
        lines.extend(
            [
                "1. [!]图像质量不满足检测要求，**建议重新采集图像**。",
                "2. 检查光照条件、相机对焦和曝光参数。",
                "3. 重新采集后再次运行质检流程。",
            ]
        )
    elif final_decision == "Suspected Defect":
        lines.extend(
            [
                "1. [*] 建议对标注的疑似区域进行**人工目视复核**。",
                "2. 结合生产工艺和缺陷历史记录评估异常的真实性。",
                "3. 如确认为真实缺陷，按质量管控流程处置。",
                "4. 如确认为误检，可调整检测参数或升级检测模型。",
            ]
        )
    else:
        lines.extend(
            [
                "1. [OK] 本次检测未发现明显异常，建议**定期抽检**。",
                "2. 如产品仍存在质量问题，可检查检测参数是否需要调整。",
                "3. 可结合其他检测手段（如 3D 点云分析）进行交叉验证。",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "*本报告由 VisualAD-Agent 自动生成，检测结果仅供参考。*",
        ]
    )

    return "\n".join(lines)


# ---------- __main__ 测试 ----------
if __name__ == "__main__":
    import argparse
    from image_quality_tool import check_image_quality
    from anomaly_detection_tool import detect_anomalies
    from visualization_tool import visualize_detection
    from vlm_tool import explain_inspection_result

    parser = argparse.ArgumentParser(description="报告生成工具")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="图像文件路径",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="请判断这张图像是否存在表面缺陷",
        help="用户检测任务描述",
    )
    args = parser.parse_args()

    # 执行完整质检流程
    print(f"[Step 1/5] 图像质量检测...")
    quality = check_image_quality(args.image)

    print(f"[Step 2/5] 异常检测...")
    anomaly = detect_anomalies(args.image)

    print(f"[Step 3/5] 生成可视化...")
    vis = visualize_detection(args.image, anomaly)

    print(f"[Step 4/5] 生成解释...")
    vlm = explain_inspection_result(args.image, args.query, quality, anomaly)

    print(f"[Step 5/5] 生成报告...")
    report = generate_inspection_report(args.image, args.query, quality, anomaly, vis, vlm)

    print(f"\n=== 最终判定: {report.final_decision} | 严重程度: {report.severity} ===\n")
    print(report.markdown_report)
