"""
骨架动作数据集与增强。

增强是骨架任务精度提升的关键 —— 骨架数据量少，容易过拟合。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore


if _HAS_TORCH:
    import torch
    from torch.utils.data import Dataset


def augment_skeleton(
    kp: np.ndarray,
    rng: np.random.Generator,
    rotate: float = 30.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    translate: float = 0.05,
    joint_dropout: float = 0.1,
    time_jitter: float = 0.1,
) -> np.ndarray:
    """
    对单段骨架做增强。

    注意：输入应当是**已归一化**的骨架（髋中心为原点，躯干长度为1），
    此时旋转/缩放/平移操作等价于对原始图像施加几何变换。

    Args:
        kp:            (T, J, 3) 归一化骨架
        rng:           随机数生成器
        rotate:        最大旋转角度（度）
        scale_range:   缩放范围
        translate:     平移幅度（相对于躯干长度）
        joint_dropout: 随机丢弃关节的比例（模拟漏检）
        time_jitter:   时间抖动比例（通过重采样改变动作速度）
    Returns:
        (T, J, 3) 增强后的骨架
    """
    out = np.asarray(kp, dtype=np.float32).copy()
    T, J, _ = out.shape

    # ── 时间抖动：改变动作速度 ────────────────────────────────────
    # ⚠️ 必须保持**输出长度不变**。
    #    定长窗口 (T=32) 是模型的输入契约,DataLoader 的 torch.stack
    #    要求 batch 内所有样本长度一致 —— 一旦这里把 T 改成 35,
    #    collate 就会报 "stack expects each tensor to be equal size"。
    #
    #    做法:先按 scale 重采样出更短/更长的序列,再插值回原长度。
    #    净效果 = 动作的**速度**变了(快一点/慢一点),
    #    但帧数和窗口边界对齐关系不变。
    if time_jitter > 0 and T > 2:
        scale = 1.0 + rng.normal(0, time_jitter)
        scale = float(np.clip(scale, 0.6, 1.6))
        new_T = max(2, int(round(T * scale)))
        src = np.linspace(0, T - 1, T)
        dst = np.linspace(0, new_T - 1, new_T)
        resampled = np.empty((new_T, J, 3), dtype=np.float32)
        for j in range(J):
            for c in range(3):
                resampled[:, j, c] = np.interp(dst, src, kp[:, j, c])
        # 插值回原长度
        back = np.linspace(0, new_T - 1, T)
        out = np.empty((T, J, 3), dtype=np.float32)
        for j in range(J):
            for c in range(3):
                out[:, j, c] = np.interp(back, dst, resampled[:, j, c])

    # ── 几何变换 ───────────────────────────────────────────────────
    # ⚠️ 注意轴顺序: out 的形状是 (T, J, 3) = (帧, 关节, (x,y,conf))。
    #    所以 x/y 坐标在**最后一维的前两个**,不是 out[:, :2]。
    #    写成 out[:, :2] 会切到"关节维",形状变成 (T,2,3),
    #    与 2×2 旋转矩阵做 matmul 直接报维度不匹配。
    ang = np.deg2rad(rng.uniform(-rotate, rotate))
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)

    sc = rng.uniform(*scale_range)
    # 只旋转 x/y,不碰 conf 通道
    xy = out[:, :, :2] @ rot.T * sc          # (T, J, 2)
    out[:, :, :2] = xy
    out[:, :, 0] += rng.uniform(-translate, translate)
    out[:, :, 1] += rng.uniform(-translate, translate)

    # ── 关节随机丢弃（模拟漏检）───────────────────────────────────
    if joint_dropout > 0:
        n_drop = max(1, int(round(joint_dropout * J)))
        idx = rng.choice(J, size=n_drop, replace=False)
        out[:, idx, 2] = 0.0

    return out


def windowize(seq: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """
    把长序列切成定长窗口。

    Args:
        seq:    (T, J, C)
        window: 窗口长度
        stride: 步长
    Returns:
        (N, window, J, C)
    """
    T = seq.shape[0]
    if T < window:
        # 不足一个窗口时，重复最后一帧补齐
        pad = np.repeat(seq[-1:], window - T, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
        T = seq.shape[0]

    n = (T - window) // stride + 1
    return np.stack([seq[i * stride : i * stride + window] for i in range(n)])


if _HAS_TORCH:

    class SkeletonActionDataset(Dataset):
        """
        骨架动作数据集。

        支持两类数据源：
          1. NTU 的 .skeleton 文件
          2. 预提取的 npz 骨架文件（key='poses'）

        每个样本返回:
            {"skel": (T, J, 3) float32, "label": int}
        """

        def __init__(
            self,
            samples: list[tuple[str, int, int]],
            num_classes: int = 60,
            window: int = 32,
            stride: int = 16,
            augment: bool = True,
            normalize: bool = True,
            seed: int = 0,
            use_coco17: bool = True,
        ) -> None:
            """
            Args:
                samples:      [(路径, 类别id, subject_id), ...]
                num_classes:  类别数
                window:       窗口长度
                stride:       切窗步长
                augment:      是否做数据增强
                normalize:    是否做身体坐标系归一化
                seed:         随机种子
                use_coco17:   是否把 NTU 25 关节转成 COCO 17
            """
            self.samples = samples
            self.num_classes = num_classes
            self.window = window
            self.stride = stride
            self.augment = augment
            self.normalize = normalize
            self.use_coco17 = use_coco17
            self.seed = seed
            self._cache: dict[int, np.ndarray] = {}

        def __len__(self) -> int:
            return len(self.samples)

        def _load(self, path: str, label: int) -> np.ndarray:
            """加载一个样本，返回 (N, window, J, 3) 的窗口序列。"""
            p = Path(path)
            if p.suffix == ".npz":
                data = np.load(p)
                seq = data["poses"]
            else:
                from actionflow.data.ntu import load_ntu_skeleton, ntu25_to_coco17

                seq = load_ntu_skeleton(p)
                if self.use_coco17:
                    seq = ntu25_to_coco17(seq)

            if self.normalize:
                from actionflow.postprocess.normalize import normalize_sequence

                seq = normalize_sequence(seq, mode="torso", rotate_to_canonical=True)

            windows = windowize(seq, self.window, self.stride)
            return windows

        def __getitem__(self, idx: int) -> dict:
            path, label, _ = self.samples[idx]
            windows = self._load(path, label)

            if len(windows) == 0:
                skel = np.zeros((self.window, 17, 3), dtype=np.float32)
                return {"skel": torch.from_numpy(skel), "label": label}

            rng = np.random.default_rng(self.seed + idx)
            if self.augment:
                # 随机选一个窗口并增强
                w = windows[rng.integers(len(windows))]
                w = augment_skeleton(w, rng)
            else:
                w = windows[0]

            return {"skel": torch.from_numpy(np.ascontiguousarray(w)), "label": label}

    class NTUActionDataset(SkeletonActionDataset):
        """NTU RGB+D 60 数据集（便捷封装）。"""

        def __init__(self, root: str | Path, split: str = "train", **kwargs) -> None:
            from actionflow.data.ntu import ntu_train_test_split, scan_ntu_dataset

            samples = scan_ntu_dataset(root)
            train, test = ntu_train_test_split(samples)
            chosen = train if split == "train" else test
            kwargs.setdefault("num_classes", 60)
            super().__init__(chosen, **kwargs)

    class AnnotatedSkeletonDataset(Dataset):
        """
        自建标注数据集 —— **带帧级标签**。

        为什么需要单独一个类
        ------------------
        前面那个 ``SkeletonActionDataset`` 每个样本只有一个标签
        ("这整段是抬手")。

        但流式模型和边界头需要的是**帧级标签**:
        - 流式模型:每一帧分别对答案
        - 边界头:每一帧到最近边界的距离

        帧级标签从哪来?从**动作段的标注**推出来:
            标注说 [30, 70) 是"抬手"
            → 第 30~69 帧的帧级标签都是"抬手"

        一个窗口可能横跨多个段,那每一帧按它所属的段来标。

        数据来源
        --------
        ``actionflow.data.annotate.export_to_npz_dataset`` 导出的 npz:
            每个文件含 ``skel`` (T,17,3)、``label``、``start_frame``、``end_frame``

        返回
        ----
            {"skel": (T, J, 3) float32,
             "label": int,              # 窗口的主标签(出现最多的那个)
             "frame_labels": (T,) int64 # 每一帧的标签
            }
        """

        def __init__(
            self,
            npz_dir: str | Path,
            label_to_id: dict[str, int],
            window: int = 32,
            stride: int = 16,
            augment: bool = True,
            normalize: bool = True,
            seed: int = 0,
        ) -> None:
            self.npz_dir = Path(npz_dir)
            self.label_to_id = label_to_id
            self.window = window
            self.stride = stride
            self.augment = augment
            self.normalize = normalize
            self.seed = seed
            self.files = sorted(self.npz_dir.glob("*.npz"))

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int) -> dict:
            data = np.load(self.files[idx])
            skel = np.asarray(data["skel"], dtype=np.float32)
            label_name = str(data["label"])

            # 帧级标签:整个窗口都是同一个(导出的窗口不跨段)
            lid = self.label_to_id.get(label_name, 0)
            frame_labels = np.full(skel.shape[0], lid, dtype=np.int64)

            if self.normalize:
                from actionflow.postprocess.normalize import normalize_sequence

                skel = normalize_sequence(skel, mode="torso", rotate_to_canonical=True)

            rng = np.random.default_rng(self.seed + idx)
            if self.augment:
                skel = augment_skeleton(skel, rng)

            return {
                "skel": torch.from_numpy(np.ascontiguousarray(skel)),
                "label": lid,
                "frame_labels": torch.from_numpy(frame_labels),
            }

else:  # pragma: no cover

    class SkeletonActionDataset:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("训练需要安装 torch：pip install 'actionflow[train]'")


def collate_batch(batch: list[dict]) -> dict:
    """
    DataLoader 的 collate_fn。

    除了骨架和段标签,如果有帧级标签(frame_labels)也会一起打包。
    没有帧级标签的数据集也能用 —— 这时只是少一个 key。

    ``segment_bounds`` 是**变长**的(每个样本覆盖的段数不同),
    所以用 list 传递,不打包成张量。
    """
    import torch

    skel = torch.stack([b["skel"] for b in batch])
    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    out = {"skel": skel, "label": label}

    # 帧级标签(可选)
    if "frame_labels" in batch[0]:
        out["frame_labels"] = torch.stack([b["frame_labels"] for b in batch])

    # 绝对动作段边界(可选)—— 边界回归头的训练目标
    if "segment_bounds" in batch[0]:
        out["segment_bounds"] = [b["segment_bounds"] for b in batch]

    # 窗口的绝对起始帧号 —— 把绝对段边界换算成窗口内相对坐标
    if "window_start" in batch[0]:
        out["window_start"] = [b["window_start"] for b in batch]

    return out
