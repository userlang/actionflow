"""
ActionFlow Pipeline — 端到端推理管线
=====================================

把三层串成一条流水线：

    视频流
      ↓  Layer 1  RTMPose (ONNX)          → 每帧 17 关键点
      ↓  后处理    One-Euro Filter        → 去抖
      ↓  后处理    身体坐标系归一化         → 位置/尺度/朝向不变
      ↓  Layer 2  SkeletonTransformer     → per-frame 动作概率
      ↓  Layer 3  Hysteresis + Viterbi    → 干净的动作段
      ↓
    [{"start":0.0, "end":1.2, "label":"左抬腿", "confidence":0.94}, ...]

三种使用方式
------------
1. **离线整段**（先算完再解码，时间戳最准）
       pipeline = ActionFlowPipeline(cfg, model)
       segments = pipeline.process_video("demo.mp4")

2. **流式滑窗**（逐段输出，延迟 ≈ window/fps）
       for seg in pipeline.process_stream("demo.mp4"):
           print(seg)      # 边跑边出

3. **纯姿态**（调试用，不需要模型）
       kps = pipeline.extract_poses("demo.mp4")
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np

from actionflow.core.config import ActionFlowConfig
from actionflow.io.pose_infer import RTMPoseInfer
from actionflow.io.video import VideoSource, open_source
from actionflow.postprocess.decode import (
    ActionSegment,
    decode_segments,
    refine_boundaries_with_regression,
)
from actionflow.postprocess.filters import SkeletonFilter
from actionflow.postprocess.normalize import normalize_keypoints


class ActionFlowPipeline:
    """
    ActionFlow 端到端管线。

    Args:
        cfg:       总配置
        model:     已训练好的分类器（可为 None，仅跑姿态）
        pose:      已加载的姿态推理器（可为 None，延迟创建）
    """

    def __init__(
        self,
        cfg: ActionFlowConfig | None = None,
        model=None,
        pose: RTMPoseInfer | None = None,
    ) -> None:
        self.cfg = cfg or ActionFlowConfig()
        self.model = model
        self._pose = pose
        self._filter = SkeletonFilter(
            min_cutoff=self.cfg.filt.min_cutoff,
            beta=self.cfg.filt.beta,
            d_cutoff=self.cfg.filt.d_cutoff,
        )
        self._labels = self.cfg.labels or []
        # process_frame 每帧更新:滤波后、归一化前的**像素坐标**骨架
        self.last_raw_pose: np.ndarray | None = None

    # ------------------------------------------------------------------
    @property
    def pose(self) -> RTMPoseInfer:
        """延迟加载姿态模型。"""
        if self._pose is None:
            self._pose = RTMPoseInfer(self.cfg.pose.onnx_path, self.cfg.pose)
        return self._pose

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        """
        单帧处理：姿态 → 滤波 → 归一化

        Args:
            frame: (H, W, 3) BGR
        Returns:
            (17, 3) 归一化后的关键点

        副作用
        ------
        把「滤波后、归一化前」的**像素坐标**存到 ``self.last_raw_pose``。
        画骨架到画面上必须用像素坐标 —— 归一化后的数值在 ±2 之间,
        int() 之后全挤在画面左上角 (0,0)。可视化路径直接读这个字段,
        不必为了一幅干净的画面把姿态模型跑两遍。
        """
        kp = self.pose(frame)

        if self.cfg.filt.enabled:
            kp = self._filter(kp, dt=dt)

        self.last_raw_pose = kp

        if self.cfg.norm.enabled:
            kp = normalize_keypoints(
                kp,
                mode=self.cfg.norm.torso_scale,
                rotate_to_canonical=self.cfg.norm.rotate_to_canonical,
                min_torso_length=self.cfg.norm.min_torso_length,
            )

        return kp

    # ------------------------------------------------------------------
    def extract_poses(
        self,
        source: str | Path | int | VideoSource,
        max_frames: int | None = None,
    ) -> np.ndarray:
        """
        提取整段视频的骨架序列。

        Returns:
            (T, 17, 3) 骨架;cfg.norm.enabled 时为身体坐标,
            关闭时为原图像素坐标(标注工具链走这条)。
        """
        src = source if isinstance(source, VideoSource) else open_source(source)
        dt = 1.0 / max(src.fps, 1e-6)

        poses = []
        for frame, _ts in src.iter_frames():
            poses.append(self.process_frame(frame, dt=dt))
            if max_frames and len(poses) >= max_frames:
                break
        src.release()

        if not poses:
            return np.zeros((0, 17, 3), dtype=np.float32)
        return np.stack(poses).astype(np.float32)

    # ------------------------------------------------------------------
    def classify_sequence(self, poses: np.ndarray) -> np.ndarray:
        """
        对骨架序列做逐帧分类。

        Args:
            poses: (T, 17, 3)
        Returns:
            (T, C) 每帧每类的概率
        """
        probs, _ = self.classify_with_boundary(poses)
        return probs

    # ------------------------------------------------------------------
    def classify_with_boundary(
        self,
        poses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        逐帧分类,**并且**顺便算出边界回归的预测。

        两者的区别
        ----------
            普通分类器 (CausalStreamingClassifier)
                只有 forward_probs(skel) → (B, T, C)

            带边界头的分类器 (BoundaryAwareClassifier)
                forward_probs(skel) 返回 (probs, boundary_dist) 两个东西

        这里统一处理:是哪种模型都能调,边界不存在时返回 None。

        Args:
            poses: (T, 17, 3)
        Returns:
            (probs, boundary_dist)
                probs:        (T, C) 每帧每类的概率
                boundary_dist: (T, 2) 每帧到最近起止边界的距离,或 None
        """
        import torch

        if self.model is None:
            raise RuntimeError("未加载分类器模型，无法执行分类")

        self.model.eval()

        # ⚠️ 必须滑窗切片,不能整段视频一次喂进去。
        #    模型的时间嵌入是 nn.Embedding(max_window),训练时 T 恒为
        #    window(32);整段视频动辄上千帧,torch.arange(T) 直接
        #    IndexError: index out of range。推理按训练同款窗口滑,
        #    重叠区取平均 —— 流式架构有前缀不变性,这样拼是合法的。
        T_total = poses.shape[0]
        win = int(self.cfg.model.window)
        stride = max(win // 2, 1)
        starts = list(range(0, max(T_total - win, 0) + 1, stride)) or [0]

        probs_sum: np.ndarray | None = None
        boundary_sum: np.ndarray | None = None
        count = np.zeros(T_total, dtype=np.float64)

        with torch.no_grad():
            for s in starts:
                seg = poses[s : s + win]
                n = seg.shape[0]  # 实际覆盖的帧数(末窗可能不足)
                if n < win:
                    # 不足一窗时补最后一帧,与训练侧 windowize 的补齐方式一致
                    pad = np.repeat(seg[-1:], win - n, axis=0)
                    seg = np.concatenate([seg, pad], axis=0)
                skel = torch.from_numpy(np.ascontiguousarray(seg)).float().unsqueeze(0)

                # 带边界头的模型返回两个值,普通模型返回一个
                out = self.model.forward_probs(skel)
                if isinstance(out, tuple):
                    p, b = out
                    p_np = p[0].cpu().numpy()
                    b_np = b[0].cpu().numpy() if b is not None else None
                else:
                    p_np = out[0].cpu().numpy()
                    b_np = None

                if probs_sum is None:
                    probs_sum = np.zeros((T_total, p_np.shape[1]), dtype=np.float64)
                probs_sum[s : s + n] += p_np[:n]
                if b_np is not None:
                    if boundary_sum is None:
                        boundary_sum = np.zeros((T_total, b_np.shape[1]), dtype=np.float64)
                    boundary_sum[s : s + n] += b_np[:n]
                count[s : s + n] += 1

        probs = probs_sum / np.maximum(count[:, None], 1.0)
        boundary_np = (
            boundary_sum / np.maximum(count[:, None], 1.0)
            if boundary_sum is not None
            else None
        )
        return probs.astype(np.float32), boundary_np

    # ------------------------------------------------------------------
    def decode(
        self,
        probs: np.ndarray,
        fps: float = 30.0,
        boundary_dist: np.ndarray | None = None,
    ) -> list[ActionSegment]:
        """
        Layer 3：把逐帧概率解码成动作段。

        如果提供了边界回归的预测,会在 Hysteresis 切段之后
        用它**微调**边界(见 refine_boundaries_with_regression)。
        """
        d = self.cfg.decode
        segments = decode_segments(
            probs,
            fps=fps,
            high=d.high_threshold,
            low=d.low_threshold,
            min_duration=d.min_duration,
            use_viterbi=d.use_viterbi,
            stay_prob=d.stay_prob,
            switch_prob=d.switch_prob,
            merge_gap=d.merge_gap,
        )

        # ── 边界回归修正 ──────────────────────────────────────────
        if boundary_dist is not None and segments:
            segments = refine_boundaries_with_regression(
                segments,
                boundary_dist,
                fps=fps,
                max_adjust=d.max_boundary_adjust,
                confidence=probs,
            )

        return segments

    # ------------------------------------------------------------------
    def process_video(
        self,
        source: str | Path | int | VideoSource,
    ) -> list[ActionSegment]:
        """
        端到端处理整段视频（离线模式）。

        时间戳最准，但需要等整段处理完才输出。
        """
        src = source if isinstance(source, VideoSource) else open_source(source)
        poses = self.extract_poses(src)
        src.release()

        if len(poses) == 0:
            return []
        if self.model is None:
            return []

        probs, boundary = self.classify_with_boundary(poses)
        return self.decode(probs, fps=src.fps, boundary_dist=boundary)

    # ------------------------------------------------------------------
    def process_stream(
        self,
        source: str | Path | int | VideoSource,
        window: int | None = None,
    ):
        """
        流式滑窗处理 —— 边跑边输出动作段。

        每次新帧到达就重算窗口内所有帧的概率，只输出"已稳定"的新增段。
        延迟 ≈ window / fps。

        Yields:
            ActionSegment
        """
        src = source if isinstance(source, VideoSource) else open_source(source)
        dt = 1.0 / max(src.fps, 1e-6)
        win = window or self.cfg.model.window

        buf = deque(maxlen=win)
        emitted: list[ActionSegment] = []

        for frame, _ts in src.iter_frames():
            kp = self.process_frame(frame, dt=dt)
            buf.append(kp)

            if len(buf) < win:
                continue
            if self.model is None:
                continue

            import torch

            skel = torch.from_numpy(np.stack(buf)).float().unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                out = self.model.forward_probs(skel)
            if isinstance(out, tuple):
                probs_t, boundary_t = out
                boundary = boundary_t[0].cpu().numpy() if boundary_t is not None else None
            else:
                probs_t = out
                boundary = None
            probs = probs_t[0].cpu().numpy()

            # 只解码到倒数第 2 帧，最后一帧还在变动
            segs = self.decode(
                probs[:-1],
                fps=src.fps,
                boundary_dist=None if boundary is None else boundary[:-1],
            )

            # 产出新增的段
            for seg in segs:
                key = (round(seg.start_frame, 3), round(seg.end_frame, 3), seg.label_id)
                if key not in {(round(s.start_frame, 3), round(s.end_frame, 3), s.label_id) for s in emitted}:
                    emitted.append(seg)
                    yield seg

        src.release()


# ===========================================================================
# 计时工具
# ===========================================================================
class Timer:
    """简单的耗时统计。"""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        pass

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0.0) + (now - self._t0)
        self._t0 = now

    def report(self) -> str:
        total = sum(self.stages.values()) or 1e-9
        lines = [f"总耗时: {total*1000:.1f} ms"]
        for k, v in self.stages.items():
            lines.append(f"  {k:20s} {v*1000:8.2f} ms  ({v/total*100:5.1f}%)")
        return "\n".join(lines)
