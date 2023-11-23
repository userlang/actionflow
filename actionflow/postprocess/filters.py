"""
One-Euro Filter — 自适应低通滤波器
==================================

出自 Casiez et al. 2012, "1€ Filter: A Simple Speed-based Low-pass Filter"
(CHI 会议论文)。已被 Google MediaPipe 官方采用。

它解决什么问题
--------------
RTMPose 逐帧独立检测姿态。同一帧里人完全静止时，检测到的关键点坐标
仍会抖动 ±2~3 像素。后果是"骨长"这一本该恒定的量在剧烈抖动：

    骨长 = |左腕 - 左肘| = 30.2, 28.9, 31.5, 29.4, ...

分类模型看到的因此是"带噪声的骨架"，训练与推理的输入分布不一致。

核心思想
--------
普通低通滤波用固定的平滑强度，会陷入两难：
  - 滤得狠 → 静止时干净，但快速挥手时有明显拖影延迟
  - 滤得轻 → 跟得上快速运动，但静止时噪声压不下去

One-Euro 的做法是让**截止频率随运动速度自适应**：
  - 关键点移动慢 → 截止频率低 → 强滤波 → 去抖
  - 关键点移动快 → 截止频率高 → 弱滤波 → 不延迟

数学
----
    dx[t]        = (x[t] - x[t-1]) / dt                  速度估计
    dx_hat[t]    = a_d * dx[t] + (1 - a_d) * dx_hat[t-1]  速度平滑
    cutoff[t]    = min_cutoff + beta * |dx_hat[t]|        自适应截止频率
    alpha[t]     = 1 / (1 + tau / teq),  tau = 1/(2*pi*cutoff)
    y[t]         = alpha[t] * x[t] + (1 - alpha[t]) * y[t-1]
"""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """标量版 One-Euro Filter（单个数值通道）。"""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_prev", "_dx_prev")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        """
        Args:
            min_cutoff: 最小截止频率（Hz）。越小 → 平滑越强、延迟越大。
            beta:       速度耦合系数。越大 → 快速运动时越敢"少滤波"。
            d_cutoff:   速度估计本身的截止频率，一般保持 1.0。
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """由截止频率和采样间隔计算平滑系数 alpha ∈ (0, 1)。"""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        teq = dt + 1.0 / (2.0 * math.pi * 1.0)  # 速度估计的等效时间常数
        return 1.0 / (1.0 + tau / teq)

    def __call__(self, x: float, dt: float = 1.0 / 30.0) -> float:
        """处理一个新观测值，返回滤波后的值。"""
        if self._x_prev is None:
            # 第一帧没有历史，直接返回
            self._x_prev = float(x)
            self._dx_prev = 0.0
            return self._x_prev

        # 1) 估计速度
        dx = (float(x) - self._x_prev) / dt

        # 2) 平滑速度
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx_prev = a_d * dx + (1.0 - a_d) * self._dx_prev

        # 3) 自适应截止频率
        cutoff = self.min_cutoff + self.beta * abs(self._dx_prev)

        # 4) 平滑当前观测
        a = self._alpha(cutoff, dt)
        out = a * float(x) + (1.0 - a) * self._x_prev

        self._x_prev = out
        return out

    def reset(self) -> None:
        """清空历史状态（用于视频分段或摄像头重连）。"""
        self._x_prev = None
        self._dx_prev = 0.0


class OneEuroFilterBank:
    """
    多通道 One-Euro Filter 组合器。

    对一个 (K,) 形状的向量（如 17 个关键点的 x 坐标）分别滤波，
    通道之间互不影响，但共享同一组参数。
    """

    def __init__(
        self,
        num_channels: int,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.num_channels = num_channels
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(num_channels)
        ]

    def __call__(self, x: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        """Args: x shape (K,) → 输出 shape (K,)"""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        for i, f in enumerate(self._filters):
            out[i] = f(x[i], dt)
        return out

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


class SkeletonFilter:
    """
    骨架专用的滤波封装：对 (T, J, C) 序列的每个通道独立滤波。

    置信度通道（index 2）通常不做滤波——它本身是模型输出的稳定分数，
    滤波会引入不必要的滞后。仅在 conf_filter=True 时开启。
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        conf_filter: bool = False,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.conf_filter = conf_filter
        # 每个关节一个 x 滤波器和 y 滤波器
        self._num_joints = 0
        self._x_bank: OneEuroFilterBank | None = None
        self._y_bank: OneEuroFilterBank | None = None

    def set_num_joints(self, num_joints: int) -> None:
        if num_joints != self._num_joints:
            self._num_joints = num_joints
            self._x_bank = OneEuroFilterBank(num_joints, self.min_cutoff, self.beta, self.d_cutoff)
            self._y_bank = OneEuroFilterBank(num_joints, self.min_cutoff, self.beta, self.d_cutoff)

    def __call__(self, keypoints: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        """
        Args:
            keypoints: (J, 3) 或 (J, 2)，最后一维为 x, y[, conf]
        Returns:
            同形状的滤波结果
        """
        kp = np.asarray(keypoints, dtype=np.float64)
        self.set_num_joints(kp.shape[0])
        assert self._x_bank is not None and self._y_bank is not None

        out = kp.copy()
        out[:, 0] = self._x_bank(kp[:, 0], dt)
        out[:, 1] = self._y_bank(kp[:, 1], dt)
        return out

    def reset(self) -> None:
        if self._x_bank is not None:
            self._x_bank.reset()
        if self._y_bank is not None:
            self._y_bank.reset()
