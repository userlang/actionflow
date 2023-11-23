"""
NTU RGB+D 数据集读取。

NTU 是骨架动作识别的标准 benchmark，60 类动作，直接提供 3D 关节坐标，
省去了从视频提骨架这一步 —— 训练时可先用它跑通 baseline。

数据格式（.skeleton 文件，每行一帧）:
    20 1 785.1 398.5 1 500.0 100.0 1 ...
    ↑帧数 ↑身体数  然后每个关节: x y z depth rotation

许可: NTU RGB+D 仅限非商业研究用途。**不要把数据打包进仓库**，
     自行下载后在 README 中说明来源。
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

# NTU 60 个动作类别
NTU60_CLASSES: list[str] = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up",
    "throw", "sit down", "stand up", "clapping", "reading", "writing",
    "tear up paper", "wear jacket", "take off jacket", "wear a shoe",
    "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping", "jump up", "phone call", "play with phone/tablet",
    "type on a keyboard", "point to something", "taking a selfie", "check time (from watch)",
    "rub two hands together", "nod head/bow", "shake head", "wipe face",
    "salute", "put palms together", "cross hands in front", "sneeze/cough",
    "staggering", "falling down", "headache", "chest pain", "back pain",
    "neck pain", "nausea/vomiting", "fan self", "punch/slap other person",
    "kicking other person", "pushing other person", "pat on back of other person",
    "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket",
    "handshake", "walking towards each other", "walking apart",
]

# 骨架中的关节数量（NTU 用 25 个关节，非 COCO 17）
NTU_NUM_JOINTS = 25

# NTU 25 关节索引 → COCO 17 关节索引 的映射（用于统一到 COCO 骨架）
# None 表示该 NTU 关节在 COCO 中没有对应点
NTU25_TO_COCO17: list[int | None] = [
    None,   # 0  base of spine
    None,   # 1  middle of spine
    None,   # 2  neck
    0,      # 3  head            → nose
    None,   # 4  left shoulder
    None,   # 5  right shoulder
    5,      # 6  left arm        → left_shoulder
    6,      # 7  right arm       → right_shoulder
    7,      # 8  left elbow
    8,      # 9  right elbow
    9,      # 10 left hand
    10,     # 11 right hand
    None,   # 12 left hip
    None,   # 13 right hip
    11,     # 14 left leg       → left_hip
    12,     # 15 right leg      → right_hip
    13,     # 16 left knee
    14,     # 17 right knee
    15,     # 18 left ankle
    16,     # 19 right ankle
    1,      # 20 left eye
    2,      # 21 right eye
    3,      # 22 left ear
    4,      # 23 right ear
    None,   # 24 left hand tip
    None,   # 25 right hand tip
]


def load_ntu_skeleton(path: str | Path) -> np.ndarray:
    """
    解析 .skeleton 文件。

    Returns:
        (T, 25, 3) 最后一维为 x, y, confidence
    """
    frames: list[list[list[float]]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            n_joints = int(parts[1])
            if n_joints != NTU_NUM_JOINTS:
                continue

            pts: list[list[float]] = []
            for j in range(n_joints):
                base = 2 + j * 3
                x = float(parts[base])
                y = float(parts[base + 1])
                depth = float(parts[base + 2])
                # NTU 用 depth==0 表示该点无效
                conf = 1.0 if depth > 0 else 0.0
                pts.append([x, y, conf])
            frames.append(pts)

    if not frames:
        return np.zeros((0, NTU_NUM_JOINTS, 3), dtype=np.float32)
    return np.array(frames, dtype=np.float32)


def ntu25_to_coco17(ntu: np.ndarray) -> np.ndarray:
    """
    NTU 25 关节 → COCO 17 关节。

    NTU 没有直接对应的关节（如"脊柱底部"）用插值估计：
      left_shoulder  = 平均(neck, left_arm)
      right_shoulder = 平均(neck, right_arm)
      left_hip       = 平均(spine_base, left_leg)
      right_hip      = 平均(spine_base, right_leg)

    Args:
        ntu: (T, 25, 3)
    Returns:
        (T, 17, 3)
    """
    ntu = np.asarray(ntu, dtype=np.float32)
    T = ntu.shape[0]
    out = np.zeros((T, 17, 3), dtype=np.float32)

    # 直接映射的关节
    for ntu_idx, coco_idx in enumerate(NTU25_TO_COCO17):
        if coco_idx is not None:
            out[:, coco_idx] = ntu[:, ntu_idx]

    # 插值出的关节
    neck, spine_base = ntu[:, 2], ntu[:, 0]
    left_arm, right_arm = ntu[:, 6], ntu[:, 7]
    left_leg, right_leg = ntu[:, 14], ntu[:, 15]

    out[:, 5] = (neck + left_arm) / 2.0    # left_shoulder
    out[:, 6] = (neck + right_arm) / 2.0   # right_shoulder
    out[:, 11] = (spine_base + left_leg) / 2.0   # left_hip
    out[:, 12] = (spine_base + right_leg) / 2.0  # right_hip

    return out


def ntu25_to_coco17_xy(ntu: np.ndarray) -> np.ndarray:
    """
    NTU 25 关节 → COCO 17 关节，只保留 x, y。

    用于把 NTU 的 3D 投影成 2D，与 RTMPose 的输出格式对齐。
    """
    return ntu25_to_coco17(ntu)[:, :, :2]


def parse_label_from_path(path: str | Path) -> int:
    """
    从文件名解析类别 id。

    NTU 文件名形如: s001_d002_p017_c055.avi
                             ^^^^ 类别
    """
    name = Path(path).name
    try:
        parts = name.split("_")
        c_part = [p for p in parts if p.startswith("c")][0]
        return int(c_part[1:])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"无法从文件名解析类别: {name}") from exc


def parse_subject_from_path(path: str | Path) -> int:
    """
    从文件名解析 subject id（用于按人划分训练/测试集）。

    NTU 文件名形如: s001_d002_p017_c055.avi
                    ^^^^
    """
    name = Path(path).name
    try:
        parts = name.split("_")
        s_part = [p for p in parts if p.startswith("s")][0]
        return int(s_part[1:])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"无法从文件名解析 subject: {name}") from exc


def scan_ntu_dataset(root: str | Path) -> list[tuple[str, int, int]]:
    """
    扫描 NTU 数据集目录。

    Returns:
        [(skeleton_path, label_id, subject_id), ...]
    """
    root = Path(root)
    files = sorted(glob.glob(os.path.join(str(root), "**", "*.skeleton"), recursive=True))
    result = []
    for f in files:
        try:
            label = parse_label_from_path(f)
            subject = parse_subject_from_path(f)
            result.append((f, label, subject))
        except ValueError:
            continue
    return result


def ntu_train_test_split(
    samples: list[tuple[str, int, int]],
    train_subjects: set[int] | None = None,
    test_subjects: set[int] | None = None,
) -> tuple[list, list]:
    """
    **按人** 划分训练/测试集。

    这是必须的 —— 如果按视频片段随机划分，同一个人的不同片段会同时
    出现在训练和测试集里，模型学到的是"记住这个人"而不是"识别动作"，
    测试精度会虚高好几个百分点。

    NTU 60 的标准划分为 subject 1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38 训练，
    其余测试。
    """
    if train_subjects is None:
        train_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
    if test_subjects is None:
        all_subjects = {s for _, _, s in samples}
        test_subjects = all_subjects - train_subjects

    train = [s for s in samples if s[2] in train_subjects]
    test = [s for s in samples if s[2] in test_subjects]
    return train, test
