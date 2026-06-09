# VisualAD-Agent

## 项目简介

**VisualAD-Agent** 是一个面向工业缺陷检测的多模态视觉质检 Agent 系统 Demo。

本项目不是单纯训练一个视觉模型，而是将视觉模型、图像处理工具、多模态大模型和报告生成模块封装为可调度的 Agent 工具链，形成一个完整的 AI 应用研发 Demo。

系统支持用户上传工业图像，自动完成图像质量检查、缺陷检测、异常区域可视化、多模态解释和结构化质检报告生成。

## 系统架构

```text
                        ┌─────────────────────────┐
                        │   Streamlit 交互页面     │
                        │   (图片上传 + 结果展示)   │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │  VisualInspectionAgent  │
                        │      (主调度器)          │
                        └───────────┬─────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        │               │           │           │               │
        ▼               ▼           ▼           ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ 图像质量检测  │ │  异常检测    │ │ 可视化   │ │ VLM 解释 │ │ 报告生成 │
│ quality_tool │ │ anomaly_tool │ │ vis_tool │ │ vlm_tool │ │report_tool│
└──────────────┘ └──────────────┘ └──────────┘ └──────────┘ └──────────┘
        │               │           │           │               │
        └───────────────┴───────────┴───────────┴───────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │   InspectionReport      │
                        │  (JSON + Markdown)       │
                        └─────────────────────────┘
```

### Agent 工作流

```text
用户上传图像 + 任务描述
         ↓
[Step 1/5] 图像质量检查 → 亮度 / 对比度 / 模糊度 / 质量等级
         ↓
[Step 2/5] 异常区域检测 → 背景残差 + 阈值分割 + 轮廓检测
         ↓
[Step 3/5] 可视化标注   → bbox + 置信度标签 + mask 叠加
         ↓
[Step 4/5] 多模态解释   → 自然语言描述异常位置、程度和建议
         ↓
[Step 5/5] 报告生成     → final_decision + severity + Markdown 报告
         ↓
   Streamlit 结果展示 + 报告下载
```

## 功能特性

### MVP 版本（已实现）

- **图像质量自动检测**：亮度均值、对比度、Laplacian 清晰度，自动判定 good/warning/poor
- **轻量异常区域检测**：基于背景残差 + 自适应阈值 + 形态学操作，不依赖深度学习权重
- **检测结果可视化**：bbox 标注、置信度标签、异常区域半透明 mask 叠加
- **结构化数据输出**：Pydantic 类型安全，6 个统一数据结构，JSON 可序列化
- **Markdown 质检报告**：含检测结论、质量详情、缺陷列表、严重程度、复核建议
- **Streamlit 交互页面**：图片上传、一键检测、双栏对比展示、报告下载

### 进阶版本（可扩展）

- 接入 YOLO / EfficientAD / PatchCore 等真实视觉模型
- 接入 SAM / SAM2 进行异常区域分割
- 接入 VLM（Qwen-VL / GPT-4o / InternVL）进行多模态解释
- 支持历史检测记录保存与查询
- 支持批量图片检测
- 支持导出 Markdown / JSON / PDF 报告
- 接入 3D 点云异常检测模型（SPA-SDF）

## 技术栈

### 当前版本

```
Python >= 3.10
Streamlit          # Web 前端
OpenCV (cv2)       # 图像处理与视觉算法
NumPy              # 数值计算
Pydantic           # 数据结构定义与校验
Matplotlib         # 可视化（预留）
Pillow             # 图像格式支持
pytest             # 单元测试
```

### 可选扩展

```
Ultralytics YOLO    # 目标检测
Segment Anything    # 实例分割
OpenAI API          # 多模态大模型
FastAPI             # API 服务
SQLite              # 历史记录
```

## 环境安装

```bash
# 克隆项目后
cd VisualAD-Agent
pip install -r requirements.txt
```

## 运行方式

### 命令行模式

```bash
# 单张图片检测
python agent.py --image data/demo_images/example.jpg --query "请判断这张图像是否存在表面缺陷"

# 单独测试各工具
python tools/image_quality_tool.py --image data/demo_images/example.jpg
python tools/anomaly_detection_tool.py --image data/demo_images/example.jpg
python tools/visualization_tool.py --image data/demo_images/example.jpg
python tools/vlm_tool.py --image data/demo_images/example.jpg
python tools/report_tool.py --image data/demo_images/example.jpg
```

### Web 界面模式

```bash
streamlit run app.py
```

浏览器打开后：
1. 左侧上传 JPG/PNG 图片
2. 输入检测任务描述
3. 点击"开始检测"
4. 查看原图/可视化对比、质量指标、缺陷列表、Markdown 报告
5. 下载 Markdown 或 JSON 报告

### 运行测试

```bash
pytest tests
# 23 passed in 0.38s
```

## 项目目录结构

```text
VisualAD-Agent/
├── app.py                          # Streamlit 前端页面
├── agent.py                        # Agent 主调度器
├── config.py                       # 全局配置（路径、阈值）
├── requirements.txt                # Python 依赖清单
├── README.md
│
├── tools/                          # 工具模块
│   ├── __init__.py
│   ├── image_quality_tool.py       # 图像质量检测
│   ├── anomaly_detection_tool.py   # 异常区域检测
│   ├── visualization_tool.py       # 检测结果可视化
│   ├── vlm_tool.py                 # 多模态解释生成
│   └── report_tool.py             # 质检报告生成
│
├── schemas/                        # Pydantic 数据结构
│   ├── __init__.py
│   └── inspection_schema.py        # 6 个统一数据模型
│
├── utils/                          # 工具函数
│   ├── __init__.py
│   ├── image_io.py                 # 图像读写（BGR/RGB）
│   └── file_utils.py              # 目录创建、时间戳
│
├── data/
│   └── demo_images/                # 测试图片
│
├── outputs/
│   ├── visualizations/             # 可视化结果图
│   ├── reports/                    # Markdown 报告
│   └── records/                    # 历史检测记录
│
└── tests/                          # pytest 单元测试
    ├── __init__.py
    ├── test_quality_tool.py        # 8 个测试
    ├── test_anomaly_tool.py        # 8 个测试
    └── test_agent.py               # 7 个测试
```

## 示例输出

### Agent 命令行输出

```
==================================================
  VisualAD-Agent 视觉质检开始
  时间: 2026-06-09 09:47:29
  图像: data/demo_images/example_defect.jpg
  任务: 请判断这张图像是否存在表面缺陷
==================================================

[Step 1/5] 图像质量检查...
  -> 亮度: 158.3, 对比度: 13.4, 清晰度: 103.6, 等级: warning

[Step 2/5] 异常区域检测...
  -> 检测到 3 个疑似异常, 全局分数: 0.9139

[Step 3/5] 生成检测可视化...
  -> 可视化已保存: outputs/visualizations/example_defect_vis_*.png

[Step 4/5] 生成多模态解释...
  -> 解释已生成 (573 字符)

[Step 5/5] 生成质检报告...

==================================================
  质检完成
  最终判定: Suspected Defect
  严重程度: high
  报告已保存: outputs/reports/example_defect_report_*.md
==================================================
```

### Markdown 报告片段

```markdown
## 检测结论

| 项目 | 结果 |
|------|------|
| 最终判定 | **疑似缺陷** |
| 严重程度 | [HIGH] **高** |
| 全局异常分数 | 0.9139 |
| 疑似缺陷数量 | 3 |

## 缺陷检测详情

| 编号 | 边界框 (x, y, w, h) | 面积 (px) | 置信度 | 类型 |
|------|---------------------|-----------|--------|------|
| #1 | [94, 274, 43, 32] | 1147 | 0.90 | suspected_surface_anomaly |
| #2 | [144, 174, 212, 27] | 5340 | 0.91 | suspected_surface_anomaly |
| #3 | [395, 95, 50, 40] | 1702 | 0.71 | suspected_surface_anomaly |
```

## 核心数据模型

```python
ImageQualityResult    # brightness, contrast, blur_score, quality_level
DefectItem            # bbox, area, confidence, defect_type
AnomalyDetectionResult # has_defect, defect_count, defects[], global_anomaly_score
VisualizationResult   # visualization_path
VLMExplanationResult  # explanation
InspectionReport      # 整合以上所有 + final_decision + severity + markdown_report
```

## 后续扩展方向

### 接入 YOLO
```text
输入图像 → YOLO 推理 → bbox → confidence → defect_type → AnomalyDetectionResult
```

### 接入 SAM
```text
YOLO bbox → SAM prompt → 缺陷 mask → 面积统计 → 可视化 mask
```

### 接入 VLM
```text
原图 + 可视化图 + 检测 JSON → VLM → 缺陷解释与复核建议
```

### 接入 3D 异常检测
```text
tools/pointcloud_anomaly_tool.py    # 新增
点云输入 → 姿态搜索 → SDF 残差计算 → 异常热力图 → 3D 检测报告
```

### 历史记录
```text
outputs/records/inspection_history.jsonl
每条记录: timestamp, image_path, final_decision, severity, global_anomaly_score
```



## 项目完成标准

以下命令全部正常运行即视为第一版完成：

```bash
python -m compileall .                         # 语法检查通过
pytest tests                                    # 23 个测试通过
python agent.py --image data/demo_images/example.jpg --query "请判断"  # 命令行运行正常
streamlit run app.py                            # Web 页面可访问
```

---

*本项目为 AI 应用研发 Demo，重点展示 Agent 工程化能力，异常检测 baseline 精度不作为第一版核心指标。*
