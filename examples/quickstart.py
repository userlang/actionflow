"""
ActionFlow 快速上手示例 —— 展示各层 API 怎么用。

运行:
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.core.constants import KEYPOINT_NAMES  # noqa: E402
from actionflow.postprocess.decode import decode_segments, format_segments  # noqa: E402
from actionflow.postprocess.filters import OneEuroFilter  # noqa: E402
from actionflow.postprocess.normalize import normalize_keypoints  # noqa: E402


def demo_1_filter() -> None:
    """示例 1：One-Euro Filter —— 单个数值通道。"""
    print("\n" + "=" * 60)
    print("示例 1 · One-Euro Filter")
    print("=" * 60)

    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    signal = [100.0] * 10 + [100.0, 101.0, 103.0, 108.0, 120.0] + [135.0] * 5
    print("  原始:  " + " ".join(f"{v:6.1f}" for v in signal[:14]))
    filtered = [f(v, dt=1 / 30) for v in signal]
    print("  滤波后:" + " ".join(f"{v:6.1f}" for v in filtered[:14]))
    print("  → 静止段平滑，快速上升段跟得上")


def demo_2_normalize() -> None:
    """示例 2：坐标归一化。"""
    print("\n" + "=" * 60)
    print("示例 2 · 身体坐标系归一化")
    print("=" * 60)

    # 造一个简单骨架：肩 y=300, 髋 y=400 → 躯干长度 100
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.95
    kp[0] = [500, 250, 0.95]   # nose
    kp[5] = [470, 300, 0.95]   # left_shoulder
    kp[6] = [530, 300, 0.95]   # right_shoulder
    kp[11] = [480, 400, 0.95]  # left_hip
    kp[12] = [520, 400, 0.95]  # right_hip

    out = normalize_keypoints(kp)

    print(f"  原点（髋中心）→ ({out[11,0]:+.3f}, {out[11,1]:+.3f}) 应该接近 (0,0)")
    torso = np.linalg.norm(out[5, :2] - out[11, :2])
    print(f"  躯干长度 → {torso:.3f} 应该等于 1.000")
    print(f"  鼻的坐标 → ({out[0,0]:+.3f}, {out[0,1]:+.3f})  y 为负说明在头顶方向（+y 朝下）")


def demo_3_decode() -> None:
    """示例 3：时序解码 —— 从逐帧概率到动作段。"""
    print("\n" + "=" * 60)
    print("示例 3 · 时序解码")
    print("=" * 60)

    labels = ["站立", "抬右手", "抬左手", "抬右腿"]
    C, T, FPS = 4, 120, 30.0

    probs = np.zeros((T, C))
    truth = [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30
    rng = np.random.default_rng(0)

    for t, c in enumerate(truth):
        probs[t, c] = float(np.clip(0.9 + rng.normal(0, 0.05), 0, 1))
        rest = (1 - probs[t, c]) / (C - 1)
        for k in range(C):
            if k != c:
                probs[t, k] = rest * (0.5 + rng.random())

    # 注入噪声
    probs[35, :] = 0.05
    probs[35, 3] = 0.90
    probs[95, :] = 0.05
    probs[95, 1] = 0.85

    print("\n  注入：第35帧孤立误判、第95帧孤立假阳性\n")

    segs = decode_segments(probs, fps=FPS, high=0.7, low=0.4, min_duration=0.3, use_viterbi=True)
    for line in format_segments(segs, labels).splitlines():
        print(f"    {line}")


def demo_4_constants() -> None:
    """示例 4：关键点定义。"""
    print("\n" + "=" * 60)
    print("示例 4 · COCO 17 关键点")
    print("=" * 60)
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"    {i:2d}  {name}")


def main() -> int:
    print("\nActionFlow 快速上手示例")
    demo_1_filter()
    demo_2_normalize()
    demo_3_decode()
    demo_4_constants()
    print("\n完成。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
