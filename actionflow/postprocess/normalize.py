"""
身体坐标系归一化 (Body-Centered Coordinate Normalization)
=========================================================

这严格来说不算"算法"，而是一组坐标变换。它的目的是让模型学的是
**动作本身**，而不是画面位置、人体型态和朝向。

三个不变性
----------
1. 位置不变：人在画面左边还是右边，不该影响"抬左手"
2. 尺度不变：身高 160 还是 190，不该影响"抬手"
3. 朝向不变：侧身 90°，旋转回标准朝向后应能识别

变换步骤
--------
    步骤 1  原点   = 左右髋中点
    步骤 2  尺度   = 躯干长度 |左肩 - 左髋|（或所有骨长中位数）
    步骤 3  朝向   = 躯干向量 (左髋 - 左肩) 归一化，作为标准 y 轴
    步骤 4  输出   = R @ (p - origin) / scale

输出坐标系：
    原点 = 髋中心
    +y   = 朝下（与图像坐标系一致），所以**头顶在负 y 方向**
    +x   = 朝人的左侧
    尺度归一化后，躯干长度恒为 1.0

举例：站直的人，归一化后
    nose      ≈ (0, -1.5)
    shoulder  ≈ (0, -1.0)
    hip       ≈ (0,  0.0)
    ankle     ≈ (0, +1.8)
"""

from __future__ import annotations

import numpy as np

from actionflow.core.constants import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NUM_KEYPOINTS,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)

# 所有骨长（用于 bone_median 尺度模式）
_BONE_PAIRS: list[tuple[int, int]] = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


def _safe_unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """归一化向量，零向量时返回原值（避免除零产生 NaN）。"""
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def compute_origin(kp: np.ndarray) -> np.ndarray:
    """髋中心作为原点。kp: (J, >=2) → (2,)"""
    hip_center = (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2.0
    return hip_center


def compute_scale(kp: np.ndarray, mode: str = "torso") -> float:
    """
    计算尺度因子。

    Args:
        kp:   (J, >=2) 关键点
        mode: "torso"       → 躯干长度 |左肩 - 左髋|
              "bone_median" → 所有骨长的中位数（对遮挡更鲁棒）
    Returns:
        尺度值（像素）。调用方需自行检查是否过小。
    """
    if mode == "bone_median":
        lengths = [
            float(np.linalg.norm(kp[a, :2] - kp[b, :2]))
            for a, b in _BONE_PAIRS
        ]
        lengths = [ln for ln in lengths if ln > 1e-6]
        return float(np.median(lengths)) if lengths else 0.0

    # 默认 torso
    return float(np.linalg.norm(kp[L_SHOULDER, :2] - kp[L_HIP, :2]))


def compute_rotation(kp: np.ndarray) -> np.ndarray:
    """
    计算从原坐标系到标准坐标系的 2D 旋转矩阵。

    标准朝向：髋中心为原点，躯干方向（髋 - 肩）为 +y 轴。
    旋转矩阵只做平面旋转，不含缩放/翻转。

    Returns:
        (2, 2) 正交矩阵
    """
    torso = kp[L_HIP, :2] - kp[L_SHOULDER, :2]  # 指向上的方向
    if float(np.linalg.norm(torso)) < 1e-6:
        return np.eye(2)

    y_axis = _safe_unit(torso)
    # x 轴 = y 轴逆时针旋转 90°（在图像坐标系 y 向下时，对应人的左侧）
    x_axis = np.array([-y_axis[1], y_axis[0]])

    # 列向量分别是 x 轴和 y 轴 → R @ v 得到标准坐标下的 v
    return np.stack([x_axis, y_axis], axis=1)


def normalize_keypoints(
    kp: np.ndarray,
    mode: str = "torso",
    rotate_to_canonical: bool = True,
    min_torso_length: float = 1.0,
) -> np.ndarray:
    """
    对单个关键点帧做身体坐标系归一化。

    Args:
        kp:                  (J, 3) 最后一维为 x, y, conf
        mode:                尺度模式，"torso" 或 "bone_median"
        rotate_to_canonical: 是否旋转到标准朝向
        min_torso_length:    躯干长度低于此值视为检测失败，原样返回
    Returns:
        (J, 3) 归一化结果；conf 原样保留
    """
    kp = np.asarray(kp)
    if kp.shape[0] != NUM_KEYPOINTS:
        raise ValueError(f"期望 {NUM_KEYPOINTS} 个关键点，收到 {kp.shape[0]}")

    origin = compute_origin(kp)
    scale = compute_scale(kp, mode=mode)
    if scale < min_torso_length:
        # 检测不可靠，保持原样（后续靠置信度 mask 屏蔽）
        return kp.copy()

    # 平移到原点 + 缩放
    out = (kp[:, :2] - origin) / scale

    if rotate_to_canonical:
        rot = compute_rotation(kp)
        out = out @ rot.T  # 等价于 rot @ out（列向量写法）

    # ⚠️ 保持输入的 dtype。
    #    之前这里写成 np.asarray(kp, dtype=np.float64),导致归一化后
    #    的骨架变成 float64,进到模型里直接报
    #    "mat1 and mat2 must have the same dtype, but got Double and Float"。
    #    几何计算内部用 float64 精度足够,输出再转回原 dtype。
    result = np.empty_like(kp, dtype=np.result_type(kp.dtype, np.float32))
    result[:, :2] = out.astype(result.dtype, copy=False)
    if kp.shape[1] >= 3:
        result[:, 2] = kp[:, 2]
    return result


def normalize_sequence(
    kps: np.ndarray,
    mode: str = "torso",
    rotate_to_canonical: bool = True,
    min_torso_length: float = 1.0,
) -> np.ndarray:
    """
    对整段序列逐帧归一化。

    Args:
        kps: (T, J, 3)
    Returns:
        (T, J, 3)
    """
    kps = np.asarray(kps)
    # ⚠️ 保持输入 dtype(见 normalize_keypoints 的说明):
    #    归一化结果要直接喂给 torch 模型,float64 会触发
    #    "mat1 and mat2 must have the same dtype" 报错。
    out = np.empty_like(kps, dtype=np.result_type(kps.dtype, np.float32))
    for t in range(kps.shape[0]):
        out[t] = normalize_keypoints(
            kps[t],
            mode=mode,
            rotate_to_canonical=rotate_to_canonical,
            min_torso_length=min_torso_length,
        )
    return out


def denormalize_keypoints(
    kp_norm: np.ndarray,
    origin: np.ndarray,
    scale: float,
    rot: np.ndarray | None = None,
) -> np.ndarray:
    """
    逆变换：把归一化坐标还原回像素坐标。

    在可视化中很有用——可以先归一化再画，保证画出来的骨架
    与模型实际看到的输入完全一致。
    """
    out = np.asarray(kp_norm, dtype=np.float64)[:, :2].copy()
    if rot is not None:
        out = out @ rot  # rot.T 的逆是 rot
    out = out * scale + origin
    return out


def compute_frame_transform(kp: np.ndarray, mode: str = "torso") -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    一次性算出 (origin, scale, rot)，供 denormalize 使用。

    Returns:
        (origin, scale, rot)  rot 为 None 表示未做旋转
    """
    origin = compute_origin(kp)
    scale = compute_scale(kp, mode=mode)
    rot = compute_rotation(kp) if scale > 1e-6 else None
    return origin, scale, rot
