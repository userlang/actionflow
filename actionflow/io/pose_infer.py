"""
Pose Inference — RTMPose ONNX 推理封装
=====================================

Layer 1 · Pose Estimation 的执行层。

RTMPose 的输出是 **SimCC 格式**：不是一个 (x, y) 坐标，而是
两条 1D 概率分布——x 方向的分布 (W × split) 和 y 方向的分布
(H × split)。取分布的期望就还原出亚像素精度的坐标。

SimCC 的价值：把「连续值回归」变成「离散值分类」，可以用交叉熵优化，
坐标精度从约 4px 提升到约 1px 以内。这是 RTMPose 高精度的关键。

本模块把上述细节封装起来，对外只暴露一个简单的接口：

    inferer = RTMPoseInfer(onnx_path)
    keypoints = inferer(frame_bgr)      # (17, 3): x, y, confidence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from actionflow.core.config import PoseConfig
from actionflow.core.constants import NUM_KEYPOINTS


class RTMPoseInfer:
    """
    RTMPose ONNX 推理器。

    使用示例:
        inferer = RTMPoseInfer("weights/rtmpose-m.onnx")
        for frame in video_frames:
            kp = inferer(frame)              # (17, 3)
    """

    def __init__(self, onnx_path: str | Path, cfg: PoseConfig | None = None) -> None:
        import onnxruntime as ort  # 延迟导入：没装也能 import 整个包

        self.cfg = cfg or PoseConfig()
        self.onnx_path = Path(onnx_path)

        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"找不到 ONNX 权重: {self.onnx_path}\n"
                f"请从 MMPose 仓库下载 RTMPose 权重并放到 weights/ 目录。\n"
                f"参考: https://github.com/open-mmlab/mmpose"
            )

        providers = self.cfg.providers
        available = ort.get_available_providers()
        # 只保留实际可用的 provider
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers,
            sess_options=_make_sess_options(),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 输出可能是 2 个（simcc_x, simcc_y）或 1 个（已解码）
        self._multi_output = len(self.session.get_outputs()) >= 2

    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        图像预处理：BGR → RGB → resize → 归一化 → NCHW

        Args:
            frame: (H, W, 3) uint8, BGR
        Returns:
            (1, 3, H, W) float32
        """
        import cv2

        img = frame
        if self.cfg.color_order.upper() == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        w, h = self.cfg.input_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 255.0
        mean = np.array(self.cfg.input_mean, dtype=np.float32)
        std = np.array(self.cfg.input_std, dtype=np.float32)
        img = (img - mean) / std

        # HWC → CHW → NCHW
        img = img.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(img, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_simcc(
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从 SimCC 分布解码坐标与置信度。

        RTMPose 的输出形状:
            simcc_x: (B, J, W*split)   x 方向每关节一条 1D 分布
            simcc_y: (B, J, H*split)   y 方向每关节一条 1D 分布

        解码方式（SimCC 的核心）:
          1. 对分布做 softmax，得到概率
          2. 取概率的**峰值位置**作为坐标（argmax，比期望更稳）
          3. 置信度 = 两个方向峰值概率的**均值**

        ⚠️ 关于置信度公式的选择:
           常见做法是 `peak_x * peak_y`（乘积），但 SimCC 分布横跨
           192 个 bin，softmax 后的峰值概率天然偏小（0.1~0.3），
           再取乘积会压到 0.01~0.05，导致所有关键点的置信度都远低于
           常用的 0.3 阈值，mask 会把全身都屏蔽掉。

           取均值可以保住峰值的量级，同时仍然要求两个方向都明确
           （任一方向不明确时均值会被拉低）。
        """
        sx = np.asarray(simcc_x, dtype=np.float64)
        sy = np.asarray(simcc_y, dtype=np.float64)

        # (B, J, L) → (J, L)
        if sx.ndim == 3:
            sx = sx[0]
            sy = sy[0]

        J = sx.shape[0]
        coords = np.zeros((J, 2), dtype=np.float32)
        confs = np.zeros(J, dtype=np.float32)

        # 数值稳定的 softmax
        def _softmax(v: np.ndarray) -> np.ndarray:
            v = v - v.max()
            e = np.exp(v)
            return e / max(e.sum(), 1e-12)

        for k in range(J):
            px = _softmax(sx[k])
            py = _softmax(sy[k])

            ix = int(px.argmax())
            iy = int(py.argmax())

            nx = max(px.shape[0] - 1, 1)
            ny = max(py.shape[0] - 1, 1)

            coords[k, 0] = ix / nx
            coords[k, 1] = iy / ny

            # 两个方向峰值概率的均值 → 该关键点的置信度
            confs[k] = float((px[ix] + py[iy]) / 2.0)

        return coords, confs

    # ------------------------------------------------------------------
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧做姿态估计。

        Args:
            frame: (H, W, 3) uint8, BGR
        Returns:
            (17, 3) 数组，每行为 [x_pixel, y_pixel, confidence]
        """
        orig_h, orig_w = frame.shape[:2]
        blob = self.preprocess(frame)

        outputs = self.session.run(self.output_names, {self.input_name: blob})

        kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

        if self._multi_output:
            # RTMPose 风格：两条 SimCC 分布
            coords, confs = self._decode_simcc(outputs[0], outputs[1])
            kp[:, :2] = coords
            kp[:, 2] = confs
        else:
            # 已经是 (1, K, 2) 或 (1, K, 3) 形式（部分导出脚本会这样）
            out = np.asarray(outputs[0])
            if out.ndim == 3:
                out = out[0]
            n = min(out.shape[0], NUM_KEYPOINTS)
            kp[:n, :2] = out[:n, :2]
            if out.shape[1] >= 3:
                kp[:n, 2] = out[:n, 2]
            else:
                kp[:n, 2] = 1.0

        # 归一化坐标 → 原图像素坐标
        kp[:, 0] *= orig_w
        kp[:, 1] *= orig_h

        return kp


def _make_sess_options():
    """设置 ONNX Runtime 会话选项（图优化 + 多线程）。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = -1  # 自动
    so.log_severity_level = 3
    return so
