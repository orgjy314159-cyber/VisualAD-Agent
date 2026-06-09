#!/bin/bash
# VisualAD-Agent 一键验证脚本

cd "$(dirname "$0")"

echo "=========================================="
echo "  VisualAD-Agent 项目完整性检查"
echo "=========================================="
echo ""

# 1. 编译检查
echo ">>> [1/3] 编译检查..."
python -m compileall . -q 2>&1
if [ $? -eq 0 ]; then
    echo "  [OK] 全部文件编译通过"
else
    echo "  [FAIL] 编译错误"
    exit 1
fi

# 2. 测试
echo ""
echo ">>> [2/3] 运行测试..."
python -m pytest tests -q 2>&1
if [ $? -eq 0 ]; then
    echo "  [OK] 全部测试通过"
else
    echo "  [FAIL] 测试未通过"
    exit 1
fi

# 3. Agent 端到端
echo ""
echo ">>> [3/3] Agent 端到端流程..."
python agent.py --image data/demo_images/example_defect.jpg --query "请判断图像是否存在缺陷" 2>&1 | grep -E "质检完成|最终判定|严重程度"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "  [OK] Agent 端到端流程正常"
else
    echo "  [FAIL] Agent 运行异常"
    exit 1
fi

echo ""
echo "=========================================="
echo "  全部验证通过！"
echo "=========================================="
echo ""
echo "启动 Web 界面: streamlit run app.py"
