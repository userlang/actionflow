"""常量定义：COCO 17 关键点、骨架连边、关节分组。"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# COCO 17 关键点 (COCO-17 keypoint order)
# ---------------------------------------------------------------------------
KEYPOINT_NAMES: list[str] = [
    "nose",        # 0
    "left_eye",    # 1
    "right_eye",   # 2
    "left_ear",    # 3
    "right_ear",   # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)

# 便捷索引
NOSE = 0
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# ---------------------------------------------------------------------------
# 骨架连边（用于可视化）
# ---------------------------------------------------------------------------
SKELETON_EDGES: list[tuple[int, int]] = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (NOSE, L_EYE),
    (NOSE, R_EYE),
    (L_EYE, L_EAR),
    (R_EYE, R_EAR),
    (NOSE, L_SHOULDER),
    (NOSE, R_SHOULDER),
]

# ---------------------------------------------------------------------------
# 关节分组（用于特征增强 / 数据增强）
# ---------------------------------------------------------------------------
UPPER_BODY: list[int] = [
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
]
LOWER_BODY: list[int] = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
LEFT_SIDE: list[int] = [L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE]
RIGHT_SIDE: list[int] = [R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE]

# 镜像时需要交换的左右关节对
LEFT_RIGHT_PAIRS: list[tuple[int, int]] = [
    (L_EYE, R_EYE), (L_EAR, R_EAR),
    (L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
    (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE),
]
