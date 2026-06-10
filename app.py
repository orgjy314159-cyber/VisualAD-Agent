"""
VisualAD-Agent Streamlit 前端页面。

提供图像上传、检测任务描述输入、一键质检、
结果展示（原图、可视化、JSON、Markdown）和报告下载功能。
"""

import sys
import os
import tempfile
import json

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from PIL import Image

from config import load_secrets
from agent import VisualInspectionAgent
from utils.file_utils import ensure_dir

# 加载密钥（Streamlit Cloud 从 st.secrets 读取）
load_secrets()

# 页面配置
st.set_page_config(
    page_title="VisualAD-Agent",
    page_icon="🔍",
    layout="wide",
)

# 初始化输出目录
ensure_dir(os.path.join(_project_root, "outputs", "visualizations"))
ensure_dir(os.path.join(_project_root, "outputs", "reports"))
ensure_dir(os.path.join(_project_root, "outputs", "records"))

# 临时上传目录
TEMP_DIR = os.path.join(_project_root, "outputs", "temp")
ensure_dir(TEMP_DIR)


@st.cache_resource
def get_agent() -> VisualInspectionAgent:
    """获取 Agent 实例（缓存，避免重复初始化）。"""
    return VisualInspectionAgent()


def save_uploaded_image(uploaded_file) -> str:
    """保存上传的文件到临时目录，返回路径。"""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    # 使用 getvalue() 获取完整 bytes（不受文件指针位置影响）
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        raise ValueError("上传文件内容为空，请重新上传")
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    # 同时用 OpenCV 验证文件可读（imdecode 支持中文路径）
    import numpy as np
    import cv2
    test = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if test is None:
        raise ValueError(
            f"图像文件无法被 OpenCV 解码，请确认格式为 JPG/PNG。"
            f"\n文件大小: {len(file_bytes)} bytes"
        )
    return file_path


def display_quality_badge(level: str):
    """根据质量等级显示彩色标签。"""
    colors = {
        "good": "green",
        "warning": "orange",
        "poor": "red",
    }
    labels = {
        "good": "良好",
        "warning": "一般",
        "poor": "较差",
    }
    color = colors.get(level, "grey")
    label = labels.get(level, level)
    st.markdown(
        f'<span style="background-color:{color};color:white;padding:4px 12px;'
        f'border-radius:12px;font-size:14px;">{label}</span>',
        unsafe_allow_html=True,
    )


def display_decision_badge(decision: str, severity: str):
    """显示最终判定和严重程度标签。"""
    decision_colors = {
        "Pass": "green",
        "Suspected Defect": "red",
        "Need Recheck": "orange",
    }
    severity_colors = {"low": "green", "medium": "orange", "high": "red"}
    dc = decision_colors.get(decision, "grey")
    sc = severity_colors.get(severity, "grey")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div style="background-color:{dc};color:white;padding:12px;'
            f'border-radius:8px;text-align:center;font-size:16px;">'
            f'<strong>判定: {decision}</strong></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="background-color:{sc};color:white;padding:12px;'
            f'border-radius:8px;text-align:center;font-size:16px;">'
            f'<strong>严重程度: {severity.upper()}</strong></div>',
            unsafe_allow_html=True,
        )


# ==================== 页面 UI ====================

st.title("VisualAD-Agent 工业视觉质检 Agent")
st.markdown(
    "上传工业图像，Agent 自动完成图像质量检查、缺陷检测、"
    "可视化标注和质检报告生成。"
)

# --- 侧边栏 ---
with st.sidebar:
    st.header("检测设置")

    uploaded_file = st.file_uploader(
        "上传检测图像",
        type=["jpg", "jpeg", "png"],
        help="支持 JPG / JPEG / PNG 格式",
    )

    user_query = st.text_input(
        "检测任务描述",
        value="请判断这张图像是否存在表面缺陷",
        help="描述你想要检测的内容",
    )

    st.divider()

    # 检测方法选择
    st.markdown("### 检测方法")
    detection_method = st.selectbox(
        "选择方法",
        options=["baseline", "yolo"],
        format_func=lambda x: {"baseline": "OpenCV baseline（规则方法）", "yolo": "YOLOv8（深度学习模型）"}[x],
        help="baseline: 基于 OpenCV 图像处理的轻量方法\nyolo: 基于 YOLOv8 自定义训练模型",
    )

    yolo_model_path = None
    yolo_confidence = 0.25
    if detection_method == "yolo":
        from config import list_yolo_models, YOLO_MODELS_DIR

        # 1) 扫描本地模型
        available_models = list_yolo_models()
        if available_models:
            model_names = [name for name, _ in available_models]
            selected_name = st.selectbox(
                "选择模型",
                options=model_names,
                help="自动扫描 models/ 目录下的 .pt 文件",
            )
            yolo_model_path = dict(available_models)[selected_name]
            st.caption(f"路径: {yolo_model_path}")

        # 2) 本地没有模型 → 允许上传（云端部署场景）
        if not available_models:
            uploaded_model = st.file_uploader(
                "上传 YOLO 模型 (.pt)",
                type=["pt"],
                help="本地无模型文件，请上传 .pt 模型",
            )
            if uploaded_model is not None:
                os.makedirs(YOLO_MODELS_DIR, exist_ok=True)
                yolo_model_path = os.path.join(YOLO_MODELS_DIR, uploaded_model.name)
                with open(yolo_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                st.success(f"模型已加载: {uploaded_model.name} ({len(uploaded_model.getbuffer()) / 1024 / 1024:.1f} MB)")
            else:
                st.info("请上传 .pt 模型文件，或切换到 baseline 方法")

        yolo_confidence = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="低于此置信度的检测框将被过滤",
        )

    st.divider()

    # VLM API 开关
    st.markdown("### AI 解释增强")
    use_vlm_api = st.checkbox(
        "启用 VLM 多模态分析",
        value=False,
        help="调用 GPT-4o / Qwen-VL 生成专业质检分析。"
        "\n需设置环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。"
        "\n未设置 API Key 时将自动回退到规则模板。",
    )
    if use_vlm_api:
        import os as _os
        has_key = _os.environ.get("DASHSCOPE_API_KEY") or _os.environ.get("OPENAI_API_KEY")
        if not has_key:
            st.warning("未检测到 API Key 环境变量，将自动回退规则模板。")
        else:
            st.success("已检测到 API Key，将使用 VLM 生成分析。")

    detect_btn = st.button(
        "开始检测",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.divider()

    st.markdown("### 关于")
    st.markdown(
        "**VisualAD-Agent** 是一个面向工业缺陷检测的"
        "多模态视觉质检 Agent Demo。"
        "将视觉模型、图像处理工具和报告生成模块封装为"
        "可调度的 Agent 工具链。"
    )

    st.markdown("### 检测流程")
    st.markdown(
        """
        1. 图像质量检查
        2. 异常区域检测
        3. 可视化标注
        4. 多模态解释
        5. 报告生成
        """
    )

# --- 主内容区 ---

if uploaded_file is None:
    # 无图片时的欢迎页面
    st.info("请在左侧上传一张工业图像开始检测。")
    st.markdown("### 系统支持")
    st.markdown("- 图像质量自动评估（亮度 / 对比度 / 模糊度）")
    st.markdown("- 轻量异常区域检测（无需深度学习权重）")
    st.markdown("- 检测结果可视化标注")
    st.markdown("- 结构化 JSON + Markdown 质检报告")

elif not detect_btn:
    # 已上传但未点击检测 — 预览图片
    st.subheader("图像预览")
    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    st.markdown("请在侧边栏点击 **开始检测** 启动质检流程。")

else:
    # 开始检测
    try:
        image_path = save_uploaded_image(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Agent 正在执行质检流程，请稍候..."):
        try:
            agent = get_agent()
            report = agent.run(
                image_path,
                user_query,
                detection_method=detection_method,
                yolo_model_path=yolo_model_path,
                yolo_confidence=yolo_confidence,
                use_vlm_api=use_vlm_api,
            )
        except Exception as e:
            # 递归获取最底层的错误信息
            cause = e
            while cause.__cause__ is not None:
                cause = cause.__cause__
            # 取根本原因的第一行（去掉 traceback）
            root_msg = str(cause).split("\n")[0]
            st.error(f"质检失败: {root_msg}")
            st.stop()

    st.success(f"质检完成 — 最终判定: **{report.final_decision}**")

    # --- 结果展示区 ---
    st.divider()

    # 判定与严重程度
    display_decision_badge(report.final_decision, report.severity)

    st.divider()

    # 图像对比：原图 vs 可视化
    st.subheader("检测结果可视化")
    col_img, col_vis = st.columns(2)
    with col_img:
        st.markdown("**原始图像**")
        original = Image.open(image_path)
        st.image(original, use_container_width=True)

    with col_vis:
        st.markdown("**检测标注**")
        vis_path = report.visualization_result.visualization_path
        if vis_path and os.path.exists(vis_path):
            vis_img = Image.open(vis_path)
            st.image(vis_img, use_container_width=True)
        else:
            st.warning("可视化结果不可用")

    st.divider()

    # 图像质量
    st.subheader("图像质量")
    q = report.quality_result
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        st.metric("亮度", f"{q.brightness:.1f}")
    with q_col2:
        st.metric("对比度", f"{q.contrast:.1f}")
    with q_col3:
        st.metric("清晰度", f"{q.blur_score:.1f}")
    with q_col4:
        st.markdown("**质量等级**")
        display_quality_badge(q.quality_level)
    st.caption(q.message)

    st.divider()

    # 异常检测
    st.subheader("缺陷检测")
    a = report.anomaly_result
    if a.has_defect:
        st.markdown(
            f"检测到 **{a.defect_count}** 个疑似异常区域，"
            f"全局异常分数 **{a.global_anomaly_score:.4f}**"
        )
        if a.defects:
            # 转换为 DataFrame 显示
            import pandas as pd

            df_data = []
            for i, d in enumerate(a.defects):
                x, y, w, h = d.bbox
                df_data.append(
                    {
                        "编号": f"#{i + 1}",
                        "位置 (x,y,w,h)": f"[{x}, {y}, {w}, {h}]",
                        "面积 (px)": f"{d.area:.0f}",
                        "置信度": f"{d.confidence:.2f}",
                        "类型": d.defect_type,
                    }
                )
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
    else:
        st.info("未检测到明显异常区域。")

    st.divider()

    # Markdown 报告
    st.subheader("质检报告")
    st.markdown(report.markdown_report)

    st.divider()

    # 下载区
    st.subheader("下载")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="下载 Markdown 报告",
            data=report.markdown_report,
            file_name=f"inspection_report_{os.path.basename(image_path)}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_dl2:
        # JSON 下载
        report_json = report.model_dump_json(indent=2)
        st.download_button(
            label="下载 JSON 结果",
            data=report_json,
            file_name=f"inspection_result_{os.path.basename(image_path)}.json",
            mime="application/json",
            use_container_width=True,
        )

    # --- 展开查看 JSON ---
    with st.expander("查看完整 JSON 结果"):
        st.json(json.loads(report_json))
