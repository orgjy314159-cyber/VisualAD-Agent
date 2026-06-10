"""
VisualAD-Agent 全局配置模块。

管理项目中的路径、阈值等可配置参数。
自动加载 .env 文件中的环境变量。
"""

import os
from pathlib import Path

# ---------- 加载 .env（本地）或 st.secrets（Streamlit Cloud） ----------
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # python-dotenv 未安装时手动解析 .env
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)
                    _val = _val.strip().strip('"').strip("'")
                    os.environ.setdefault(_key.strip(), _val)


def load_secrets():
    """加载密钥（兼容本地 .env 和 Streamlit Cloud st.secrets）。

    调用时机：app.py 启动时（此时 streamlit 已可用）。
    本地：已在模块加载时从 .env 读取，此函数为空操作。
    云端：从 st.secrets 读取并写入 os.environ。
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            for key in ["SOPHNET_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"]:
                if key in st.secrets and not os.environ.get(key):
                    os.environ[key] = st.secrets[key]
    except Exception:
        pass  # 非 streamlit 环境，忽略

# ---------- 项目根目录 ----------
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

# ---------- 路径配置 ----------
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
DEMO_IMAGES_DIR: str = os.path.join(DATA_DIR, "demo_images")
OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
VISUALIZATIONS_DIR: str = os.path.join(OUTPUTS_DIR, "visualizations")
REPORTS_DIR: str = os.path.join(OUTPUTS_DIR, "reports")
RECORDS_DIR: str = os.path.join(OUTPUTS_DIR, "records")

# ---------- 图像质量阈值 ----------
BLUR_THRESHOLD: float = 80.0       # Laplacian 方差低于此值判定为模糊
DARK_THRESHOLD: float = 40.0       # 亮度均值低于此值判定为过暗
BRIGHT_THRESHOLD: float = 220.0    # 亮度均值高于此值判定为过曝

# ---------- 异常检测参数 ----------
MIN_CONTOUR_AREA: int = 100        # 最小轮廓面积，过滤噪声
GAUSSIAN_KERNEL: int = 5           # 高斯滤波核大小
ADAPTIVE_THRESH_BLOCK: int = 11    # 自适应阈值邻域大小

# ---------- 检测方法 ----------
DETECTION_METHOD: str = "baseline"  # 默认检测方法: "baseline" | "yolo"
YOLO_MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
YOLO_MODEL_PATH: str = os.path.join(YOLO_MODELS_DIR, "black.pt")
YOLO_CONFIDENCE: float = 0.25       # YOLO 置信度阈值


def list_yolo_models() -> list:
    """扫描 models 目录，返回所有 .pt 文件列表。

    Returns:
        list[tuple]: [(模型文件名, 完整路径), ...]，按文件名排序。
    """
    if not os.path.isdir(YOLO_MODELS_DIR):
        return []
    models = []
    for f in os.listdir(YOLO_MODELS_DIR):
        if f.endswith(".pt"):
            models.append((f, os.path.join(YOLO_MODELS_DIR, f)))
    return sorted(models, key=lambda x: x[0])
