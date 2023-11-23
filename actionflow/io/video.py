"""
视频读写：文件 / 摄像头 / 目录三种输入源的统一接口。

统一成同一个 `iter_frames()` 生成器，上层 pipeline 不需要关心来源。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


class VideoSource:
    """
    视频源基类。

    统一接口:
        for frame, timestamp in source.iter_frames():
            ...
    """

    def __init__(self) -> None:
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0

    def iter_frames(self) -> Iterator[tuple[np.ndarray, float]]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoFileSource(VideoSource):
    """从视频文件读取。"""

    def __init__(self, path: str | Path) -> None:
        import cv2

        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"视频不存在: {self.path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def iter_frames(self) -> Iterator[tuple[np.ndarray, float]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame, idx / self.fps
            idx += 1

    def release(self) -> None:
        self.cap.release()


class CameraSource(VideoSource):
    """从摄像头/实时流读取。"""

    def __init__(self, device: int = 0, width: int = 1280, height: int = 720) -> None:
        import cv2

        super().__init__()
        self.device = device
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def iter_frames(self) -> Iterator[tuple[np.ndarray, float]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                # 短暂读取失败（USB 抖动）时跳过，不要中断整个流
                continue
            yield frame, idx / self.fps
            idx += 1

    def release(self) -> None:
        self.cap.release()


class ImageDirSource(VideoSource):
    """从图片目录读取（当作视频处理，常用于数据集可视化）。"""

    def __init__(self, directory: str | Path, pattern: str = "*.jpg", fps: float = 30.0) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.fps = fps
        files = sorted(self.directory.glob(pattern))
        if not files:
            raise FileNotFoundError(f"{self.directory} 下没有匹配 {pattern} 的文件")
        self.files = files
        self.frame_count = len(files)

    def iter_frames(self) -> Iterator[tuple[np.ndarray, float]]:
        import cv2

        for idx, f in enumerate(self.files):
            img = cv2.imread(str(f))
            if img is None:
                continue
            self.height, self.width = img.shape[:2]
            yield img, idx / self.fps

    def release(self) -> None:
        pass


def open_source(src: str | Path | int) -> VideoSource:
    """
    自动判断输入源类型并打开。

    Args:
        src: 视频文件路径 / 图片目录 / 摄像头编号(int)
    """
    if isinstance(src, int):
        return CameraSource(src)
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.is_dir():
            return ImageDirSource(p)
        return VideoFileSource(p)
    raise TypeError(f"不支持的输入源类型: {type(src)}")
