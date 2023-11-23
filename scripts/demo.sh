#!/usr/bin/env bash
# ActionFlow 快速演示 —— 一键跑完全部检查
#
# 用法:
#   bash scripts/demo.sh
#
# 不需要 ONNX 权重、不需要训练数据。

set -euo pipefail

PY="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           ActionFlow — 流水线演示                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "▶ [1/5] 生成合成测试视频..."
$PY tools/make_demo_video.py --out assets/demo.mp4 --seconds 8

echo ""
echo "▶ [2/5] 流水线自检（滤波 / 归一化 / 分类器 / 解码）..."
$PY tools/verify_pipeline.py

echo ""
echo "▶ [3/5] 解码算法演示（三种病症 → 三种解法）..."
$PY -m actionflow.cli decode-test

echo ""
echo "▶ [4/5] 四种架构演示（流式 / 边界回归 / 双路融合）..."
$PY examples/architectures.py

echo ""
echo "▶ [5/5] 单元测试..."
FAILED=0
for t in tests/test_*.py; do
    printf "  %-26s " "$(basename "$t")"
    if OUT=$($PY "$t" 2>/dev/null); then
        echo "$OUT" | grep -E "通过" || echo "失败"
    else
        echo "失败"
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "  ✗ 有测试失败"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✓ 全部完成"
echo ""
echo "  下一步:"
echo "    1. 下载真实 RTMPose 权重到 weights/rtmpose-m.onnx"
echo "    2. python -m actionflow.cli pose --video assets/demo.mp4 --visualize out.mp4"
echo "    3. 训练分类器: python -m actionflow.train --data /path/to/nturgbd"
echo "════════════════════════════════════════════════════════════"
echo ""
