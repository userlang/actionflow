"""
生成一段合成的测试视频 —— 不需要下载任何数据集就能验证整条流水线。

合成内容：一个简单的人形（用关键点连成的骨架）在画面中做几种动作，
        叠加高斯噪声，模拟 RTMPose 的检测抖动。

用法:
    python tools/make_demo_video.py --out assets/demo.mp4 --seconds 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_keyframe(t: float, action: int) -> np.ndarray:
    """
    生成某一时刻的 17 个关键点（COCO 17）。

    动作定义:
        0 站立
        1 抬右手
        2 抬左手
        3 抬右腿
    """
    cx, cy = 320.0, 360.0

    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.95

    # 基础站姿
    shoulder_y, hip_y = cy - 60.0, cy + 20.0
    l_sh = np.array([cx - 45.0, shoulder_y])
    r_sh = np.array([cx + 45.0, shoulder_y])
    l_hip = np.array([cx - 30.0, hip_y])
    r_hip = np.array([cx + 30.0, hip_y])

    l_el = np.array([cx - 55.0, cy - 10.0])
    r_el = np.array([cx + 55.0, cy - 10.0])
    l_wr = np.array([cx - 58.0, cy + 45.0])
    r_wr = np.array([cx + 58.0, cy + 45.0])
    l_kn = np.array([cx - 32.0, cy + 100.0])
    r_kn = np.array([cx + 32.0, cy + 100.0])
    l_an = np.array([cx - 33.0, cy + 180.0])
    r_an = np.array([cx + 33.0, cy + 180.0])

    # 动作 1: 抬右手 → 右臂向上
    if action == 1:
        r_el = np.array([cx + 75.0, cy - 90.0])
        r_wr = np.array([cx + 85.0, cy - 165.0])

    # 动作 2: 抬左手 → 左臂向上
    if action == 2:
        l_el = np.array([cx - 75.0, cy - 90.0])
        l_wr = np.array([cx - 85.0, cy - 165.0])

    # 动作 3: 抬右腿 → 右膝抬高
    if action == 3:
        r_kn = np.array([cx + 60.0, cy + 40.0])
        r_an = np.array([cx + 70.0, cy - 20.0])

    for idx, pt in [
        (0, (cx, cy - 105.0)),      # nose
        (1, (cx - 8, cy - 112.0)),  # left_eye
        (2, (cx + 8, cy - 112.0)),  # right_eye
        (3, (cx - 18, cy - 108.0)), # left_ear
        (4, (cx + 18, cy - 108.0)), # right_ear
        (5, l_sh), (6, r_sh),
        (7, l_el), (8, r_el),
        (9, l_wr), (10, r_wr),
        (11, l_hip), (12, r_hip),
        (13, l_kn), (14, r_kn),
        (15, l_an), (16, r_an),
    ]:
        kp[idx, :2] = pt

    return kp


def draw_skeleton_frame(
    img: np.ndarray,
    kp: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """在图上画一个简单人形（躯干 + 四肢）。"""
    import cv2

    edges = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 5), (0, 6),
    ]
    for i, j in edges:
        p1 = (int(kp[i, 0]), int(kp[i, 1]))
        p2 = (int(kp[j, 0]), int(kp[j, 1]))
        cv2.line(img, p1, p2, color, 4, cv2.LINE_AA)
    for idx in range(17):
        pt = (int(kp[idx, 0]), int(kp[idx, 1]))
        cv2.circle(img, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def main() -> int:
    import cv2

    p = argparse.ArgumentParser(description="生成合成测试视频")
    p.add_argument("--out", default="assets/demo.mp4")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--noise", type=float, default=3.0, help="关键点噪声标准差（像素）")
    args = p.parse_args()

    n_frames = int(args.seconds * args.fps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )

    # 动作时间线: 每 2 秒换一个动作
    actions = [0, 1, 2, 3]
    rng = np.random.default_rng(0)

    for i in range(n_frames):
        t = i / args.fps
        action = actions[min(int(t // 2.0), len(actions) - 1)]

        # 动作之间用平滑过渡，避免跳变
        phase = (t % 2.0) / 2.0
        blend = min(1.0, phase / 0.15)  # 前 15% 时间做过渡

        kp = make_keyframe(t, action)
        if phase < 0.15 and action > 0:
            kp_prev = make_keyframe(t, actions[max(0, action - 1)])
            kp = kp_prev * (1 - blend) + kp * blend

        # 叠加噪声（模拟检测抖动）
        kp_noisy = kp.copy()
        kp_noisy[:, :2] += rng.normal(0, args.noise, size=(17, 2))

        img = np.full((args.height, args.width, 3), 40, dtype=np.uint8)
        draw_skeleton_frame(img, kp_noisy)

        cv2.putText(
            img, f"t={t:.1f}s  action={action}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(img)

    writer.release()
    print(f"✓ 已生成: {out_path}  ({n_frames} 帧, {args.fps}fps, {args.seconds}s)")
    print("  动作时间线: 0-2s 站立 | 2-4s 抬右手 | 4-6s 抬左手 | 6-8s 抬右腿")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
