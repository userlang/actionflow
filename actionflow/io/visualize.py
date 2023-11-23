"""
可视化：骨架绘制 + 动作段叠加。

用于调试和 demo 视频生成。
"""

from __future__ import annotations

import numpy as np

from actionflow.core.constants import KEYPOINT_NAMES, SKELETON_EDGES

# BGR 配色
COLORS = {
    "skeleton": (0, 255, 0),
    "skeleton_filtered": (0, 200, 255),
    "text": (255, 255, 255),
    "hud_bg": (30, 30, 30),
    "segment": (0, 140, 255),
}

# 不同置信度用不同颜色：绿=高, 黄=中, 红=低
def _conf_color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.7:
        return (0, 220, 0)
    if conf >= 0.4:
        return (0, 200, 255)
    return (0, 0, 255)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.3,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    radius: int = 4,
    draw_names: bool = False,
) -> np.ndarray:
    """
    在帧上绘制骨架。

    Args:
        frame:        (H, W, 3) BGR，会被原地修改（返回同一对象）
        keypoints:   (17, 3) x, y, conf
        conf_threshold: 低于此置信度的关节不画
        color:       统一颜色；None 则按置信度渐变
    Returns:
        frame（原地修改）
    """
    import cv2

    if color is None:
        color = COLORS["skeleton"]

    kp = np.asarray(keypoints)
    for i, j in SKELETON_EDGES:
        if kp[i, 2] < conf_threshold or kp[j, 2] < conf_threshold:
            continue
        pt1 = (int(kp[i, 0]), int(kp[i, 1]))
        pt2 = (int(kp[j, 0]), int(kp[j, 1]))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    for idx in range(len(kp)):
        if kp[idx, 2] < conf_threshold:
            continue
        pt = (int(kp[idx, 0]), int(kp[idx, 1]))
        c = color if color is not None else _conf_color(float(kp[idx, 2]))
        cv2.circle(frame, pt, radius, c, -1, cv2.LINE_AA)
        if draw_names:
            cv2.putText(
                frame, KEYPOINT_NAMES[idx], (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["text"], 1, cv2.LINE_AA,
            )

    return frame


def draw_hud(
    frame: np.ndarray,
    lines: list[str],
    position: tuple[int, int] = (15, 30),
    font_scale: float = 0.6,
) -> np.ndarray:
    """绘制左上角信息面板。"""
    import cv2

    x, y = position
    line_h = 22
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLORS["text"], 1, cv2.LINE_AA,
        )
    return frame


def draw_action_bar(
    frame: np.ndarray,
    segments: list,
    total_duration: float,
    current_time: float,
    labels: list[str] | None = None,
    bar_height: int = 26,
) -> np.ndarray:
    """
    在画面底部画一条动作时间轴条。

    每个动作段用不同颜色表示，当前播放位置用白线标记。
    """
    import cv2

    if not segments or total_duration <= 0:
        return frame

    h, w = frame.shape[:2]
    y0 = h - bar_height - 8
    x0, x1 = 10, w - 10
    total_w = x1 - x0

    cv2.rectangle(frame, (x0, y0), (x1, y0 + bar_height), COLORS["hud_bg"], -1)

    palette = [
        (255, 180, 60), (120, 220, 255), (180, 255, 140),
        (255, 140, 180), (160, 160, 255), (255, 220, 120),
    ]

    for seg in segments:
        sx = x0 + int(seg.start_time / total_duration * total_w)
        ex = x0 + int(seg.end_time / total_duration * total_w)
        ex = max(ex, sx + 3)
        c = palette[seg.label_id % len(palette)]
        cv2.rectangle(frame, (sx, y0 + 2), (ex, y0 + bar_height - 2), c, -1)

        name = labels[seg.label_id] if labels and seg.label_id < len(labels) else str(seg.label_id)
        if ex - sx > 30:
            cv2.putText(
                frame, name[:6], (sx + 3, y0 + bar_height - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA,
            )

    # 当前播放位置
    cx = x0 + int(min(current_time / total_duration, 1.0) * total_w)
    cv2.line(frame, (cx, y0 - 3), (cx, y0 + bar_height + 3), (255, 255, 255), 2)

    return frame


def make_side_byside(frames_a: list[np.ndarray], frames_b: list[np.ndarray]) -> list[np.ndarray]:
    """把两组帧并排拼接（用于滤波前 vs 滤波后对比）。"""
    out = []
    for a, b in zip(frames_a, frames_b):
        h = max(a.shape[0], b.shape[0])
        w = a.shape[1] + b.shape[1]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: a.shape[0], : a.shape[1]] = a
        canvas[: b.shape[0], a.shape[1] :] = b
        out.append(canvas)
    return out
