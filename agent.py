"""
VisualAD-Agent 主调度器模块。

VisualInspectionAgent 按照质检工作流依次调度各个工具：
图像质量 → 异常检测 → 可视化 → 解释 → 报告生成。
"""

import sys
import os
import argparse
import traceback
from datetime import datetime

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.image_quality_tool import check_image_quality
from tools.anomaly_detection_tool import detect_anomalies
from tools.visualization_tool import visualize_detection
from tools.vlm_tool import explain_inspection_result
from tools.report_tool import generate_inspection_report
from schemas.inspection_schema import InspectionReport
from utils.file_utils import ensure_dir

# 报告输出目录
_REPORT_OUTPUT_DIR = os.path.join(_project_root, "outputs", "reports")


class VisualInspectionAgent:
    """视觉质检 Agent 主调度器。

    按照可控工作流依次执行质检流程中的各个步骤，
    收集中间结果并生成最终质检报告。

    使用示例:
        agent = VisualInspectionAgent()
        report = agent.run("path/to/image.jpg", "检查表面缺陷")
        print(report.markdown_report)
    """

    def __init__(self):
        """初始化 Agent，确保输出目录存在。"""
        ensure_dir(_REPORT_OUTPUT_DIR)

    def run(
        self,
        image_path: str,
        user_query: str = "",
        detection_method: str = "baseline",
        yolo_model_path: str = None,
        yolo_confidence: float = 0.25,
        use_vlm_api: bool = False,
    ) -> InspectionReport:
        """执行完整的视觉质检流程。

        流程步骤：
        1. 图像质量检查
        2. 异常区域检测（可选 baseline 或 yolo）
        3. 检测结果可视化
        4. 多模态解释生成（可选 VLM API 或规则模板）
        5. 最终报告生成

        Args:
            image_path: 待检测的图像文件路径。
            user_query: 用户输入的检测任务描述。
            detection_method: 检测方法 "baseline" | "yolo"。
            yolo_model_path: YOLO 模型 .pt 文件路径。
            yolo_confidence: YOLO 置信度阈值。
            use_vlm_api: 是否调用真实 VLM API 生成解释。

        Returns:
            InspectionReport: 完整的质检报告。

        Raises:
            FileNotFoundError: 图像文件不存在。
            RuntimeError: 质检流程中某个步骤执行失败。
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        if not user_query:
            user_query = "请对该图像进行工业视觉质检"

        print(f"{'=' * 50}")
        print(f"  VisualAD-Agent 视觉质检开始")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  图像: {image_path}")
        print(f"  任务: {user_query}")
        print(f"{'=' * 50}")

        try:
            # Step 1: 图像质量检查
            print("\n[Step 1/5] 图像质量检查...")
            quality_result = check_image_quality(image_path)
            print(
                f"  -> 亮度: {quality_result.brightness:.1f}, "
                f"对比度: {quality_result.contrast:.1f}, "
                f"清晰度: {quality_result.blur_score:.1f}, "
                f"等级: {quality_result.quality_level}"
            )

            # Step 2: 异常检测
            method_label = "YOLOv8" if detection_method == "yolo" else "baseline"
            print(f"\n[Step 2/5] 异常区域检测（{method_label}）...")
            anomaly_result = detect_anomalies(
                image_path,
                method=detection_method,
                model_path=yolo_model_path,
                yolo_conf=yolo_confidence,
            )
            print(
                f"  -> 检测到 {anomaly_result.defect_count} 个疑似异常, "
                f"全局分数: {anomaly_result.global_anomaly_score:.4f}"
            )

            # Step 3: 可视化
            print("\n[Step 3/5] 生成检测可视化...")
            vis_result = visualize_detection(image_path, anomaly_result)
            print(f"  -> 可视化已保存: {vis_result.visualization_path}")

            # Step 4: 解释生成
            print("\n[Step 4/5] 生成多模态解释...")
            vlm_result = explain_inspection_result(
                image_path, user_query, quality_result, anomaly_result,
                use_api=use_vlm_api, vis_image_path=vis_result.visualization_path,
            )
            print(f"  -> 解释已生成 ({len(vlm_result.explanation)} 字符)")

            # Step 5: 报告生成
            print("\n[Step 5/5] 生成质检报告...")
            report = generate_inspection_report(
                image_path,
                user_query,
                quality_result,
                anomaly_result,
                vis_result,
                vlm_result,
            )

            # 保存报告到文件
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(_REPORT_OUTPUT_DIR, f"{image_stem}_report_{ts}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report.markdown_report)

            print(f"\n{'=' * 50}")
            print(f"  质检完成")
            print(f"  最终判定: {report.final_decision}")
            print(f"  严重程度: {report.severity}")
            print(f"  报告已保存: {report_path}")
            print(f"{'=' * 50}")

            return report

        except FileNotFoundError:
            raise
        except Exception as e:
            error_msg = (
                f"质检流程执行失败: {str(e)}\n\n"
                f"详细错误:\n{traceback.format_exc()}"
            )
            print(f"\n[ERROR] {error_msg}")
            raise RuntimeError(error_msg) from e


# ---------- __main__ ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VisualAD-Agent 视觉质检 Agent"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="待检测的图像文件路径",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="请判断这张图像是否存在表面缺陷",
        help="用户检测任务描述",
    )
    parser.add_argument(
        "--detection-method",
        type=str,
        default="baseline",
        choices=["baseline", "yolo"],
        help="检测方法: baseline | yolo",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="YOLO .pt 模型路径（--detection-method yolo 时生效）",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO 置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="启用真实 VLM API（需设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量）",
    )
    args = parser.parse_args()

    agent = VisualInspectionAgent()
    report = agent.run(
        args.image,
        args.query,
        detection_method=args.detection_method,
        yolo_model_path=args.yolo_model,
        yolo_confidence=args.yolo_conf,
        use_vlm_api=args.use_vlm,
    )

    # 打印 Markdown 报告
    print("\n" + "=" * 50)
    print("  最终质检报告")
    print("=" * 50)
    print(report.markdown_report)
