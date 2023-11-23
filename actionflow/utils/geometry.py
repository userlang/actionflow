"""几何与骨架相关的通用工具。"""

from __future__ import annotations

import numpy as np

from actionflow.core.constants import (
    LEFT_RIGHT_PAIRS,
    NUM_KEYPOINTS,
)


def bone_lengths(kp: np.ndarray, pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    """计算所有骨长。kp: (J, >=2) → (len(pairs),)"""
    if pairs is None:
        from actionflow.postprocess.normalize import _BONE_PAIRS

        pairs = _BONE_PAIRS
    kp = np.asarray(kp)
    return np.array([np.linalg.norm(kp[a, :2] - kp[b, :2]) for a, b in pairs])


def mirror_keypoints(kp: np.ndarray) -> np.ndarray:
    """
    左右镜像关键点（用于数据增强）。

    注意：镜像后 x 坐标需要翻转（相对躯干中线），仅交换左右索引是不够的。
    这里只做索引交换 —— 因为在归一化后的身体坐标系里，x 轴已经以
    髋中心为原点、指向人的左侧，镜像增强时需要额外翻转 x 符号。

    Args:
        kp: (J, >=2)
    Returns:
        同形状，交换了左右关节索引
    """
    kp = np.asarray(kp).copy()
    for a, b in LEFT_RIGHT_PAIRS:
        tmp = kp[a].copy()
        kp[a] = kp[b]
        kp[b] = tmp
    return kp


def flip_x(kp: np.ndarray) -> np.ndarray:
    """翻转 x 坐标（配合 mirror 使用）。"""
    kp = np.asarray(kp).copy()
    kp[:, 0] = -kp[:, 0]
    return kp


def keypoint_velocity(kp: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
    """
    关键点速度。

    Args:
        kp:  (T, J, >=2)
    Returns:
        (T, J, 2) 每帧的速度
    """
    kp = np.asarray(kp, dtype=np.float64)
    vel = np.zeros_like(kp[:, :, :2])
    vel[1:] = (kp[1:, :, :2] - kp[:-1, :, :2]) / dt
    return vel


def relative_positions(kp: np.ndarray) -> np.ndarray:
    """
    关节相对位置的成对特征。

    对每个关节，计算它到「身体其他关节」的相对向量。这是骨架动作识别
    中非常有效的补充特征（比绝对坐标更能表达"左手在哪"）。

    Args:
        kp: (T, J, >=2)
    Returns:
        (T, J, J-1, 2) 相对向量特征
    """
    kp = np.asarray(kp, dtype=np.float64)[:, :, :2]
    T, J, _ = kp.shape

    # (T, J, 1, 2) - (T, 1, J, 2) → (T, J, J, 2)
    diff = kp[:, :, None, :] - kp[:, None, :, :]

    # 去掉对角线（自己减自己）
    mask = ~np.eye(J, dtype=bool)
    return diff[:, :, mask, :]


def resample_sequence(kp: np.ndarray, target_len: int) -> np.ndarray:
    """
    时间重采样 —— 用于速度扰动增强。

    把 (T, J, C) 插值到 (target_len, J, C)。

    Args:
        kp:        (T, J, C)
        target_len: 目标长度
    Returns:
        (target_len, J, C)
    """
    kp = np.asarray(kp, dtype=np.float32)
    T = kp.shape[0]
    if T == target_len:
        return kp
    if T == 1:
        return np.repeat(kp, target_len, axis=0)

    src_idx = np.linspace(0, T - 1, T)
    dst_idx = np.linspace(0, T - 1, target_len)

    out = np.empty((target_len,) + kp.shape[1:], dtype=np.float32)
    for j in range(kp.shape[1]):
        for c in range(kp.shape[2]):
            out[:, j, c] = np.interp(dst_idx, src_idx, kp[:, j, c])
    return out


def random_rotate_2d(kp: np.ndarray, angle_rad: float) -> np.ndarray:
    """绕原点旋转 2D 坐标。"""
    kp = np.asarray(kp, dtype=np.float64).copy()
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    kp[:, :2] = kp[:, :2] @ rot.T
    return kp


def random_scale(kp: np.ndarray, scale: float) -> np.ndarray:
    """等比缩放 2D 坐标。"""
    kp = np.asarray(kp, dtype=np.float64).copy()
    kp[:, :2] *= scale
    return kp


def random_translate(kp: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """平移 2D 坐标。"""
    kp = np.asarray(kp, dtype=np.float64).copy()
    kp[:, 0] += dx
    kp[:, 1] += dy
    return kp


def mask_random_joints(kp: np.ndarray, ratio: float = 0.1, rng=None) -> np.ndarray:
    """
    随机遮挡关节 —— 模拟漏检。

    被遮挡的关节置信度置 0，让模型学会不依赖某些关键点。
    """
    rng = rng or np.random.default_rng()
    kp = np.asarray(kp, dtype=np.float64).copy()
    n = max(1, int(round(ratio * NUM_KEYPOINTS)))
    idx = rng.choice(NUM_KEYPOINTS, size=n, replace=False)
    kp[idx, 2] = 0.0
    return kp
