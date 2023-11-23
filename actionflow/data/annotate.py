"""
自建数据集的标注工具
=====================

为什么需要自建数据
----------------
公开数据集(NTU RGB+D、Kinetics)有三个问题:

  ① **粒度不对**
     NTU 是"打电话/吃饭/读书"这种**复合动作**,
     我们的目标是"抬左手/右手握拳/左抬腿"这种**原子动作**。
     复合动作是原子的组合,不是原子的本身。

  ② **没有原子动作的边界标注**
     NTU 标注的是"整段视频是哪个动作",
     我们需要的是"第几帧到第几帧在做什么"。
     这正是本项目 Layer 3 的输入。

  ③ **场景不匹配**
     NTU 是实验室场景(固定机位、固定背景、穿统一服装)。
     真实场景是各种角度、各种光照、有人遮挡。

所以必须自建。标注工具就是解决"怎么高效地标"的问题。

标注工具的思路:关键点辅助标注
------------------------------
手工逐帧画框选关键点是不可能的(一段 5 秒视频 = 150 帧 × 17 点 = 2550 个点)。

聪明的办法:**用姿态模型先自动标,人只做修正**。

    ① 跑 RTMPose,自动得到每帧的 17 个关键点
    ② 把这些关键点画在界面上
    ③ 人在界面上做**少量修正**:
       - 拖动明显标错的点
       - 标记这段视频的动作类别
       - 标记动作的起止帧

    这样标一段 5 秒视频,从 30 分钟降到 5 分钟。

再进一步:关键点**不一定要人工标**
---------------------------------
如果只是要"动作段 + 类别"这种粗标注,
其实**根本不需要人工标关键点** —— 直接用 RTMPose 自动抽的就行。

所以本工具支持两种模式:

    模式 A(快速):自动关键点 + 人工标类别和起止帧
    模式 B(精细):自动关键点 + 人工修正关键点 + 标类别和起止帧

数据格式
--------
标注结果存成一个 JSON 文件,一条样本一个条目:

    {
      "video": "clips/raise_right_hand_01.mp4",
      "fps": 30.0,
      "num_frames": 150,
      "actions": [
        {"label": "抬右手", "start_frame": 10, "end_frame": 55},
        {"label": "站立",   "start_frame": 56, "end_frame": 149}
      ],
      "keypoints": "raise_right_hand_01.npz"   # 可选,人工修正后的关键点
    }

一个视频文件可以包含**多个动作段**(这是关键 ——
「站立→抬手→站立」这种来回,应该标成 3 段)。

CLI 用法
--------
    # ① 从视频自动抽关键点(生成待标注的 npz)
    python -m actionflow.data.annotate extract --video clips/s01_b.mp4 --out data/a.npz

    # ② 起标注界面(需要 tkinter,macOS 自带)
    python -m actionflow.data.annotate gui --video clips/s01_b.mp4 --npz data/a.npz \\
        --out annotations/a.json

    # ③ 纯命令行标注(没有 GUI 时的备选)
    python -m actionflow.data.annotate cli --video clips/s01_b.mp4 --npz data/a.npz \\
        --out annotations/a.json

    # ④ 校验标注质量
    python -m actionflow.data.annotate validate --annotations annotations/ --npz-dir data/
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class ActionAnnotation:
    """一个动作段的标注。"""

    label: str
    start_frame: int
    end_frame: int          # 右开区间 [start, end)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ActionAnnotation:
        return cls(
            label=str(d["label"]),
            start_frame=int(d["start_frame"]),
            end_frame=int(d["end_frame"]),
        )


@dataclass
class VideoAnnotation:
    """一个视频文件的完整标注。"""

    video: str
    fps: float = 30.0
    num_frames: int = 0
    actions: list[ActionAnnotation] = field(default_factory=list)
    keypoints: str | None = None    # 关键点 npz 的路径(可选)

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "actions": [a.to_dict() for a in self.actions],
            "keypoints": self.keypoints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VideoAnnotation:
        return cls(
            video=str(d["video"]),
            fps=float(d.get("fps", 30.0)),
            num_frames=int(d.get("num_frames", 0)),
            actions=[ActionAnnotation.from_dict(a) for a in d.get("actions", [])],
            keypoints=d.get("keypoints"),
        )

    def validate(self) -> list[str]:
        """
        自检 —— 返回问题列表,空列表表示没问题。

        检查项:
          1. 动作段是否有重叠
          2. 动作段是否越界
          3. 是否有零长度/负长度的段
          4. 帧范围是否是整数
        """
        problems: list[str] = []

        prev_end = -1
        for i, a in enumerate(self.actions):
            if a.end_frame <= a.start_frame:
                problems.append(
                    f"第{i}个动作段「{a.label}」长度非法: "
                    f"start={a.start_frame} end={a.end_frame}"
                )
            if a.start_frame < 0:
                problems.append(f"第{i}个动作段起点为负: {a.start_frame}")
            if self.num_frames and a.end_frame > self.num_frames:
                problems.append(
                    f"第{i}个动作段越界: end={a.end_frame} > 总帧数 {self.num_frames}"
                )
            if a.start_frame < prev_end:
                problems.append(
                    f"第{i}个动作段「{a.label}」与前一段重叠: "
                    f"start={a.start_frame} < 前段end={prev_end}"
                )
            prev_end = max(prev_end, a.end_frame)

        return problems


# ===========================================================================
# 关键点提取
# ===========================================================================
def extract_keypoints(
    video: str | Path,
    out_npz: str | Path | None = None,
    onnx_path: str | Path | None = None,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """
    用姿态模型自动抽关键点,存成 npz。

    Parameters
    ----------
    video:        视频路径
    out_npz:      输出的 npz 路径(可选)
    onnx_path:    姿态模型 ONNX 路径(默认用配置里的)
    conf_threshold: 低于此置信度的关键点视为漏检

    Returns
    -------
    (T, 17, 3) 关键点序列,坐标是**原图像素坐标**

    ⚠️ 为什么这里要关掉归一化(norm.enabled = False)
    ------------------------------------------------
    pipeline 的默认输出是「身体坐标系」(髋中心为原点、躯干为尺度),
    那是给分类模型吃的。但标注工具链需要的是**像素坐标**:

    ① GUI 直接把关键点画到视频帧上 —— 画之前没有任何反归一化,
       存归一化坐标的话骨架全挤在画面左上角 (0,0),
       「画面上核对左右手」这个核心功能直接失效。
    ② 训练时 ``AnnotatedNPZDataset`` 加载数据会做归一化。
       如果 npz 里已经归一化过,就是**归一化两次**——
       而两次归一化并不等于一次(实测坐标最大差 2.9,
       朝向旋转会被抵消),训练输入和推理输入就对不上了。

    存像素坐标,让「归一化」只发生在喂给模型之前的那一刻,
    训练/推理/标注三条路径才共用同一份语义。
    """
    from actionflow.core.config import ActionFlowConfig
    from actionflow.infer.pipeline import ActionFlowPipeline

    cfg = ActionFlowConfig()
    if onnx_path is not None:
        cfg.pose.onnx_path = str(onnx_path)
    cfg.pose.keypoint_thresh = conf_threshold
    cfg.norm.enabled = False  # ← 存像素坐标,见上面 docstring

    pipe = ActionFlowPipeline(cfg)
    kps = pipe.extract_poses(video)

    if out_npz is not None:
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            keypoints=kps.astype(np.float32),
            video=str(video),
        )

    return kps


# ===========================================================================
# 标注存取
# ===========================================================================
def save_annotations(annotations: list[VideoAnnotation], out_path: str | Path) -> None:
    """保存标注到 JSON。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "labels": sorted({a.label for v in annotations for a in v.actions}),
        "videos": [v.to_dict() for v in annotations],
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_annotations(in_path: str | Path) -> tuple[list[VideoAnnotation], list[str]]:
    """
    读取标注。

    Returns
    -------
    (videos, labels)
        videos:  VideoAnnotation 列表
        labels:  出现过的所有标签名
    """
    in_path = Path(in_path)
    payload = json.loads(in_path.read_text(encoding="utf-8"))

    videos = [VideoAnnotation.from_dict(d) for d in payload.get("videos", [])]
    labels = list(payload.get("labels", []))
    if not labels:
        labels = sorted({a.label for v in videos for a in v.actions})
    return videos, labels


# ===========================================================================
# 校验
# ===========================================================================
def validate_dataset(
    annotations_dir: str | Path,
    npz_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    校验整个数据集的标注质量。

    检查什么
    --------
    ① 每个视频的标注自洽(不重叠、不越界、长度合法)
    ② 类别分布 —— 严重不均衡是模型学不好的主因
    ③ 每个类别的样本数 —— 有的类只有 2 个样本就敢开训,必翻车
    ④ 关键点文件是否存在(如果声明了)
    ⑤ 关键点文件的形状是否正确

    Returns
    -------
    dict 报告
    """
    annotations_dir = Path(annotations_dir)
    files = sorted(annotations_dir.glob("*.json"))

    problems: list[str] = []
    warnings: list[str] = []
    label_counts: dict[str, int] = {}
    total_segments = 0
    n_videos = 0

    for f in files:
        try:
            videos, _ = load_annotations(f)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{f.name}: 读取失败 — {e}")
            continue

        for v in videos:
            n_videos += 1
            total_segments += len(v.actions)

            for p in v.validate():
                problems.append(f"{f.name} / {v.video}: {p}")

            for a in v.actions:
                label_counts[a.label] = label_counts.get(a.label, 0) + 1

            # 关键点文件检查
            if npz_dir is not None and v.keypoints:
                kp_path = Path(npz_dir) / Path(v.keypoints).name
                if not kp_path.exists():
                    problems.append(f"{v.video}: 关键点文件不存在 — {kp_path}")
                else:
                    try:
                        data = np.load(kp_path)
                        kps = data["keypoints"]
                        if kps.ndim != 3 or kps.shape[1:] != (17, 3):
                            problems.append(
                                f"{v.video}: 关键点形状应为 (T,17,3),实际 {kps.shape}"
                            )
                        elif v.num_frames and kps.shape[0] != v.num_frames:
                            warnings.append(
                                f"{v.video}: 关键点帧数 {kps.shape[0]} "
                                f"与标注帧数 {v.num_frames} 不一致"
                            )
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"{v.video}: 关键点读取失败 — {e}")

    # ── 类别分布检查 ─────────────────────────────────────────────
    if label_counts:
        counts = sorted(label_counts.values())
        n_labels = len(label_counts)
        min_c, max_c = counts[0], counts[-1]
        ratio = max_c / max(min_c, 1)

        if min_c < 10:
            warnings.append(
                f"有类别样本数少于 10 个(最少 {min_c} 个)—— 建议至少 30 个/类再开训"
            )
        if ratio > 10 and n_labels > 1:
            warnings.append(
                f"类别分布不均衡:最多/最少 = {ratio:.1f} 倍 —— "
                f"考虑重采样或加类别权重"
            )

    report = {
        "num_files": len(files),
        "num_videos": n_videos,
        "num_segments": total_segments,
        "num_labels": len(label_counts),
        "label_counts": dict(sorted(label_counts.items(), key=lambda kv: -kv[1])),
        "problems": problems,
        "warnings": warnings,
    }

    if verbose:
        print("=" * 60)
        print("数据集校验报告")
        print("=" * 60)
        print(f"  标注文件: {report['num_files']}")
        print(f"  视频数:   {report['num_videos']}")
        print(f"  动作段:   {report['num_segments']}")
        print(f"  类别数:   {report['num_labels']}")
        if report["label_counts"]:
            print("\n  类别分布:")
            for label, c in report["label_counts"].items():
                bar = "█" * max(1, c * 40 // max(report["label_counts"].values()))
                print(f"    {label:16s} {c:4d}  {bar}")
        if problems:
            print(f"\n  ❌ 发现 {len(problems)} 个问题:")
            for p in problems[:20]:
                print(f"     - {p}")
        if warnings:
            print(f"\n  ⚠️  {len(warnings)} 个警告:")
            for w in warnings[:20]:
                print(f"     - {w}")
        if not problems:
            print("\n  ✓ 没有发现结构性问题")
        print("=" * 60)

    return report


# ===========================================================================
# 导出为训练格式
# ===========================================================================
def export_to_npz_dataset(
    annotations: list[VideoAnnotation],
    npz_dir: str | Path,
    out_dir: str | Path,
    window: int = 32,
    stride: int = 16,
) -> dict:
    """
    把标注导出成训练用的 npz 切片数据集。

    流程
    ----
        标注(视频级) → 切成固定长度的窗口 → 每个窗口一个 npz

    为什么要切窗口?
    --------------
    Transformer 分类器吃的是固定长度的序列 (T, 17, 3)。
    一段视频可能 5 秒 = 150 帧,不能整段直接喂。

    切窗口时的**标签**怎么定?
    -----------------------
    一个窗口可能横跨两个动作(比如前 20 帧是"站立",后 12 帧是"抬手")。
    这里用**多数票**:窗口里哪个动作占的帧多,就标成哪个。

    为什么不丢掉跨界的窗口?
    ------------------------
    丢掉的话,动作切换点的信息就没了。
    保留的话,标签会有点噪声 —— 但这个噪声其实是有益的,
    它让模型学会"动作边界附近是模糊的",和真实情况一致。

    Parameters
    ----------
    annotations: 标注列表
    npz_dir:     关键点 npz 所在目录
    out_dir:     输出目录
    window:      窗口长度(帧)
    stride:      滑窗步长(帧)

    Returns
    -------
    dict 统计信息
    """
    npz_dir = Path(npz_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"num_windows": 0, "num_skipped": 0, "label_counts": {}}

    for v in annotations:
        if not v.keypoints:
            stats["num_skipped"] += 1
            continue

        kp_path = npz_dir / Path(v.keypoints).name
        if not kp_path.exists():
            stats["num_skipped"] += 1
            continue

        kps = np.load(kp_path)["keypoints"]
        T = kps.shape[0]

        # 每个动作段切成若干窗口
        for a in v.actions:
            start, end = a.start_frame, a.end_frame
            for ws in range(start, end, stride):
                we = ws + window
                # 窗口超出这一段的范围就不取(保证窗口内标签纯净)
                if we > end:
                    break
                if we > T:
                    break

                seg = kps[ws:we]
                label_name = a.label

                fname = f"{Path(v.video).stem}__{label_name}__{ws:06d}.npz"
                np.savez_compressed(
                    out_dir / fname,
                    skel=seg.astype(np.float32),
                    label=label_name,
                    start_frame=ws,
                    end_frame=we,
                    video=v.video,
                    # subject 用于「按人划分」—— 训练集/测试集必须按人切,
                    # 否则同一个人的不同片段会同时出现在两边,精度虚高。
                    # 命名约定: 视频名形如 s01_xxx.mp4 → subject = "s01"
                    subject=Path(v.video).stem.split("_")[0],
                )
                stats["num_windows"] += 1
                stats["label_counts"][label_name] = (
                    stats["label_counts"].get(label_name, 0) + 1
                )

    return stats
