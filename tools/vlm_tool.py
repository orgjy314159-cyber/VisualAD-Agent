"""
多模态解释模块。

根据图像质量和异常检测结果生成自然语言质检解释。
支持三种模式（按优先级）：
1. Qwen-VL（DashScope，OpenAI 兼容接口）
2. GPT-4o（OpenAI 原生接口）
3. 规则模板（本地回退）
"""

import sys
import os
import base64
import json

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from schemas.inspection_schema import (
    ImageQualityResult,
    AnomalyDetectionResult,
    VLMExplanationResult,
)

# VLM API 配置（按优先级排列）
_SOPHNET_BASE_URL = "https://www.sophnet.com/api/open-apis/v1"
_SOPHNET_MODEL = "Qwen3.5-397B-A17B"

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_QWEN_MODEL = "qwen-vl-max"

_GPT4O_MODEL = "gpt-4o"

_API_TIMEOUT = 60
_VLM_MAX_TOKENS = 800


def explain_inspection_result(
    image_path: str,
    user_query: str,
    quality_result: ImageQualityResult,
    anomaly_result: AnomalyDetectionResult,
    use_api: bool = False,
    vis_image_path: str = None,
) -> VLMExplanationResult:
    """根据质检结果生成自然语言解释。

    Args:
        image_path: 原始图像路径。
        user_query: 用户输入的检测任务描述。
        quality_result: 图像质量检测结果。
        anomaly_result: 异常检测结果。
        use_api: 是否尝试调用 VLM API。
        vis_image_path: 可视化结果图路径（可选，发给 VLM 参考）。

    Returns:
        VLMExplanationResult: 自然语言解释结果。
    """
    if use_api:
        return _explain_via_api(
            image_path, user_query, quality_result, anomaly_result, vis_image_path
        )

    return _explain_via_rules(image_path, user_query, quality_result, anomaly_result)


# ==================== API 调用 ====================

def _explain_via_api(
    image_path: str,
    user_query: str,
    quality_result: ImageQualityResult,
    anomaly_result: AnomalyDetectionResult,
    vis_image_path: str = None,
) -> VLMExplanationResult:
    """依次尝试 DashScope Qwen-VL → OpenAI GPT-4o → 规则回退。"""

    prompt = _build_vlm_prompt(user_query, quality_result, anomaly_result)
    image_b64 = _encode_image_b64(image_path)
    vis_b64 = _encode_image_b64(vis_image_path) if vis_image_path else None

    def _build_messages():
        """构建多模态消息，有可视化图则同时发送两张图。"""
        content = [{"type": "text", "text": prompt}]
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })
        if vis_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{vis_b64}"},
            })
        return [{"role": "user", "content": content}]

    messages = _build_messages()
    # 纯文本消息（用于不支持视觉的模型）
    text_messages = [{"role": "user", "content": prompt}]

    # 1) 尝试 Sophnet Qwen
    sophnet_key = os.environ.get("SOPHNET_API_KEY")
    if sophnet_key:
        try:
            print("[INFO] 尝试 Sophnet Qwen3.5 ...")
            # 先尝试带图的多模态请求
            try:
                text = _call_openai_compatible(
                    api_key=sophnet_key,
                    base_url=_SOPHNET_BASE_URL,
                    model=_SOPHNET_MODEL,
                    messages=messages,
                )
            except Exception:
                # 如果不支持图片，回退纯文本
                print("[INFO] Sophnet 不支持图片输入，使用纯文本模式 ...")
                text = _call_openai_compatible(
                    api_key=sophnet_key,
                    base_url=_SOPHNET_BASE_URL,
                    model=_SOPHNET_MODEL,
                    messages=text_messages,
                )
            return VLMExplanationResult(explanation=text)
        except Exception as e:
            print(f"[WARN] Sophnet 调用失败: {e}")

    # 2) 尝试 DashScope Qwen-VL
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    if dashscope_key:
        try:
            print("[INFO] 尝试 DashScope Qwen-VL ...")
            text = _call_openai_compatible(
                api_key=dashscope_key,
                base_url=_DASHSCOPE_BASE_URL,
                model=_QWEN_MODEL,
                messages=messages,
            )
            return VLMExplanationResult(explanation=text)
        except Exception as e:
            print(f"[WARN] DashScope 调用失败: {e}")

    # 3) 尝试 OpenAI GPT-4o
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            print("[INFO] 尝试 OpenAI GPT-4o ...")
            text = _call_openai_compatible(
                api_key=openai_key,
                base_url=None,
                model=_GPT4O_MODEL,
                messages=messages,
            )
            return VLMExplanationResult(explanation=text)
        except Exception as e:
            print(f"[WARN] OpenAI 调用失败: {e}")

    # 4) 全部失败，回退规则模板
    print("[WARN] 所有 VLM API 均不可用，回退到规则模板。")
    return _explain_via_rules(image_path, user_query, quality_result, anomaly_result)


def _call_openai_compatible(
    api_key: str,
    base_url: str | None,
    model: str,
    messages: list,
) -> str:
    """通过 OpenAI 兼容接口调用 VLM。

    Args:
        api_key: API Key。
        base_url: 自定义 Base URL（DashScope 用），None 则用 OpenAI 默认。
        model: 模型名称。
        messages: 多模态消息列表。

    Returns:
        str: VLM 回复文本。
    """
    from openai import OpenAI

    kwargs = {"api_key": api_key, "timeout": _API_TIMEOUT}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=_VLM_MAX_TOKENS,
        temperature=0.3,
    )

    text = response.choices[0].message.content
    if text is None:
        raise ValueError("VLM 返回空响应")
    return text.strip()


# ==================== Prompt 构建 ====================

def _build_vlm_prompt(
    user_query: str,
    quality_result: ImageQualityResult,
    anomaly_result: AnomalyDetectionResult,
) -> str:
    """构建发给 VLM 的提示词（包含质检上下文）。"""

    defects_info = ""
    if anomaly_result.has_defect and anomaly_result.defects:
        defects_info = f"检测到 {anomaly_result.defect_count} 个疑似缺陷：\n"
        for i, d in enumerate(anomaly_result.defects):
            x, y, w, h = d.bbox
            defects_info += (
                f"  - 缺陷 #{i + 1}: 类型={d.defect_type}, "
                f"位置=[{x},{y},{w},{h}], 面积={d.area:.0f}px, "
                f"置信度={d.confidence:.2f}\n"
            )
        defects_info += f"全局异常分数: {anomaly_result.global_anomaly_score:.4f}\n"
    else:
        defects_info = "本次自动检测未发现明显异常区域。\n"

    quality_info = (
        f"亮度均值={quality_result.brightness:.1f}, "
        f"对比度={quality_result.contrast:.1f}, "
        f"清晰度得分={quality_result.blur_score:.1f}, "
        f"质量等级={quality_result.quality_level}"
    )

    prompt = f"""你是一名专业的工业视觉质检专家。请根据以下检测结果和图像，撰写一份专业的质检分析报告。

## 用户任务
{user_query}

## 图像质量检测结果
{quality_info}

## 自动缺陷检测结果
{defects_info}

## 要求
1. 首先评价图像质量是否影响检测结果的可信度。
2. 仔细观察图像，结合上面的检测结果，分析图像中是否存在真实缺陷。
3. 如果存在缺陷，描述缺陷的外观特征、位置和可能的原因。
4. 评估严重程度（低/中/高），给出处理建议。
5. 如果自动检测结果与图像实际表现不符（例如误检或漏检），请指出来。
6. 最后给出综合结论和人工复核建议。

请用专业但易懂的中文撰写，控制在 300-500 字。"""

    return prompt


# ==================== 图像编码 ====================

def _encode_image_b64(image_path: str) -> str:
    """将图像文件编码为 base64 字符串。

    Args:
        image_path: 图像文件路径。

    Returns:
        str: base64 编码字符串。
    """
    import numpy as np
    import cv2

    # 使用 imdecode 支持中文路径
    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 编码为 JPEG
    _, jpg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(jpg_buf.tobytes()).decode("utf-8")


# ==================== 规则模板回退 ====================

def _explain_via_rules(
    image_path: str,
    user_query: str,
    quality_result: ImageQualityResult,
    anomaly_result: AnomalyDetectionResult,
) -> VLMExplanationResult:
    """使用规则模板生成解释。"""

    image_name = os.path.basename(image_path)
    parts = []

    parts.append(f"针对用户任务「{user_query}」，系统对图像「{image_name}」进行了自动化质检分析。")

    quality_parts = []
    quality_parts.append(
        f"亮度均值 {quality_result.brightness:.1f}，对比度 {quality_result.contrast:.1f}，"
        f"清晰度得分 {quality_result.blur_score:.1f}。"
    )

    quality_desc = {
        "good": "图像质量良好，各项指标均在正常范围内，检测结果可信度较高。",
        "warning": "图像质量存在一定偏差，可能对检测精度产生轻微影响，建议在条件允许时重新采集。",
        "poor": "图像质量较差，检测结果可能不可靠，强烈建议重新采集图像后再进行检测。",
    }
    quality_parts.append(quality_desc.get(quality_result.quality_level, ""))

    issues = []
    if quality_result.is_blurry:
        issues.append("图像存在明显模糊")
    if quality_result.is_too_dark:
        issues.append("图像亮度过低（偏暗）")
    if quality_result.is_too_bright:
        issues.append("图像亮度过高（过曝）")
    if issues:
        quality_parts.append("具体问题：" + "、".join(issues) + "。")

    parts.append("【图像质量】" + " ".join(quality_parts))

    if anomaly_result.has_defect and anomaly_result.defects:
        count = anomaly_result.defect_count
        parts.append(f"【缺陷检测】系统在图像中共检测到 {count} 个疑似异常区域。")

        for i, defect in enumerate(anomaly_result.defects):
            x, y, w, h = defect.bbox
            location_desc = _describe_location(x, y, w, h)
            parts.append(
                f"  异常 #{i + 1}：位于{location_desc}，"
                f"边界框 [{x}, {y}, {w}, {h}]，"
                f"面积 {defect.area:.0f} 像素，"
                f"置信度 {defect.confidence:.2f}，"
                f"类型为 {defect.defect_type}。"
            )

        score = anomaly_result.global_anomaly_score
        if score >= 0.7:
            severity_desc = "异常程度较高"
            suggestion = "建议优先安排人工复核，必要时进行二次采样检测。"
        elif score >= 0.4:
            severity_desc = "存在一定程度的异常"
            suggestion = "建议进入人工复核流程，结合生产工艺判断是否需要进一步处理。"
        else:
            severity_desc = "异常程度较低"
            suggestion = "可结合生产上下文判断是否需要关注，建议定期抽检。"

        parts.append(f"  全局异常分数 {score:.2f}，{severity_desc}。{suggestion}")
    else:
        parts.append(
            "【缺陷检测】系统未在图像中检测到明显异常区域。"
            "图像整体表现均匀，无明显纹理突变或不连续。"
            "但仍建议结合人工目视复核，以排除微小或隐蔽缺陷的遗漏可能。"
        )

    parts.append(
        "【综合建议】本报告由自动化视觉质检 Agent 生成，"
        "检测结果仅供参考，最终判定应以人工复核为准。"
        "如检测结果与实际情况不符，可调整检测参数或接入更高精度的视觉模型。"
    )

    explanation = "\n\n".join(parts)
    return VLMExplanationResult(explanation=explanation)


def _describe_location(x: int, y: int, w: int, h: int) -> str:
    if x < 200:
        h_pos = "图像左侧"
    elif x > 400:
        h_pos = "图像右侧"
    else:
        h_pos = "图像中部"
    if y < 133:
        v_pos = "上方"
    elif y > 266:
        v_pos = "下方"
    else:
        v_pos = "中部"
    if h_pos == "图像中部" and v_pos == "中部":
        return "图像中央区域"
    return f"{h_pos}{v_pos}"


# ---------- __main__ 测试 ----------
if __name__ == "__main__":
    import argparse
    from image_quality_tool import check_image_quality
    from anomaly_detection_tool import detect_anomalies
    from visualization_tool import visualize_detection

    parser = argparse.ArgumentParser(description="VLM 解释工具")
    parser.add_argument("--image", type=str, required=True, help="图像文件路径")
    parser.add_argument(
        "--query", type=str, default="请判断这张图像是否存在表面缺陷", help="检测任务描述"
    )
    parser.add_argument("--use-api", action="store_true", help="启用 VLM API 调用")
    args = parser.parse_args()

    quality = check_image_quality(args.image)
    anomaly = detect_anomalies(args.image)

    # 生成可视化图发给 VLM
    vis_path = None
    if args.use_api:
        try:
            vis = visualize_detection(args.image, anomaly)
            vis_path = vis.visualization_path
        except Exception:
            pass

    result = explain_inspection_result(
        args.image, args.query, quality, anomaly,
        use_api=args.use_api, vis_image_path=vis_path,
    )

    print(result.explanation)
    print("\n--- VLMExplanationResult JSON ---")
    print(result.model_dump_json(indent=2))
