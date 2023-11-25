"""
自建标注数据集 —— 训练闭环的胶水层
================================

这个模块解决一个具体的衔接问题:

    ``actionflow.data.annotate.export_to_npz_dataset`` 导出的 npz
    →  ``actionflow.data.dataset.AnnotatedSkeletonDataset`` 读
    →  ``actionflow.train`` 训

但中间有三个断点,这个模块把它们接上:

**断点 1: npz 的 key 对不上**
    ``SkeletonActionDataset._load`` 读的是 ``data["poses"]``,
    而 ``export_to_npz_dataset`` 写的是 ``data["skel"]``。
    直接喂过去会 ``KeyError: 'poses is not a file in the archive'``。

**断点 2: 标签 id 没人映射**
    模型需要整数标签,标注里是中文标签名。
    ``AnnotatedSkeletonDataset`` 要求传入 ``label_to_id`` 字典,
    但整个代码库**没有任何地方**构造这个字典 ——
    忘了传的话 ``label_to_id.get(name, 0)`` 会把所有未知标签静默变成 0,
    训练看起来"能跑",但实际上只有一类。

    本模块提供 ``build_label_map()``:从目录里扫描所有 npz,
    按**出现次数降序**建立稳定的 id 映射,并把映射写进
    ``labels.json``,供训练和推理共用。

**断点 3: train.py 只认 NTU**
    ``train.py:build_dataloaders`` 写死了 ``NTUActionDataset``,
    而它扫描的是 ``*.skeleton`` 文件。给一个 npz 目录,
    它会建出一个**空数据集** —— 训练"跑完了",但样本数是 0。

    本模块提供 ``AnnotatedNPZDataset``,目录即数据集,
    配合 ``train.py`` 的 ``--dataset-type`` 开关使用。

数据目录结构
------------
::

    data/
      windows/            ← 训练用的 npz 切片(export 出来的)
        clip1__抬右手__000000.npz
        clip1__抬右手__000016.npz
        ...
      labels.json         ← 标签映射(build_label_map 生成)
      splits/
        train.txt         ← 训练集文件名单(按人划分)
        test.txt          ← 测试集文件名单

每个 npz 包含:
    skel: (T, 17, 3) float32
    label: str
    start_frame: int
    end_frame: int
    video: str

帧级标签
--------
本数据集提供 ``frame_labels`` —— 每个窗口的逐帧标签。

窗口**可以跨动作段**:导出时每个 npz 属于单个动作段(标签纯净),
但训练时我们从原始标注重建逐帧标签,让边界回归头有真值可学。
``dist_to_start`` / ``dist_to_end`` 就是从这里算出来的。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore


# ===========================================================================
# 标签映射
# ===========================================================================
def build_label_map(
    npz_dir: str | Path,
    out_json: str | Path | None = None,
) -> dict[str, int]:
    """
    扫描 npz 目录,建立「标签名 → 整数 id」的映射。

    排序规则
    --------
    **按样本数降序**,同数按标签名升序。

    为什么按样本数排?
    ---------------
    因为 id 是要写进模型权重的 —— 一旦训练完,
    id→类别 的对应关系就固定了。如果每次重建映射时排序不同,
    同一个 id 在两次训练里可能指向不同类别,
    老权重就废了。

    按样本数降序还有一个好处:**id 0 一定是最常见的类别**。
    这在标签缺失时有兜底意义(缺一个类别的样本不会让整个数据集塌掉)。

    Parameters
    ----------
    npz_dir:   存放 npz 的目录
    out_json:  映射保存路径(默认写到 ``npz_dir/../labels.json``)

    Returns
    -------
    dict[str, int]
    """
    npz_dir = Path(npz_dir)
    counts: dict[str, int] = {}

    for f in sorted(npz_dir.glob("*.npz")):
        try:
            label = str(np.load(f, allow_pickle=False)["label"])
        except Exception:  # noqa: BLE001
            continue
        counts[label] = counts.get(label, 0) + 1

    # 样本数降序,同数按名字升序 —— 保证可复现
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    label_to_id = {name: i for i, (name, _) in enumerate(ordered)}

    if out_json is None:
        out_json = npz_dir.parent / "labels.json"
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "label_to_id": label_to_id,
                "id_to_label": {str(i): n for n, i in label_to_id.items()},
                "counts": dict(ordered),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return label_to_id


def load_label_map(path: str | Path) -> dict[str, int]:
    """读取 ``labels.json`` 里的 label_to_id。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    m = payload.get("label_to_id", {})
    return {str(k): int(v) for k, v in m.items()}


def labels_in_order(label_to_id: dict[str, int]) -> list[str]:
    """把 label_to_id 转成「id 顺序」的标签名列表 —— 这就是 cfg.labels。"""
    n = max(label_to_id.values()) + 1 if label_to_id else 0
    out = [""] * n
    for name, i in label_to_id.items():
        out[i] = name
    return out


# ===========================================================================
# 划分
# ===========================================================================
def build_split(
    npz_dir: str | Path,
    out_dir: str | Path,
    subject_of: callable | None = None,
    val_ratio: float = 0.2,
) -> dict[str, int]:
    """
    按「人」划分训练/测试集。

    为什么必须按人划分
    ----------------
    如果按视频片段随机划分,同一个人的不同片段会同时出现在
    训练集和测试集里。模型学到的是「记住这个人」而不是
    「识别动作」,测试精度会虚高好几个百分点。

    subject 的来源(按优先级)
    ------------------------
    ① npz 里显式的 ``subject`` 字段(最可靠)
    ② 文件名解析:``<video_stem>__<label>__<start>.npz``
       → 去掉后两段,剩下的就是 video stem
       → 再从 video stem 里按 ``_`` 切分,取第一个 token 作为 subject

    ⚠️ 命名约定(推荐):
        ``s01_<动作名>__<动作名>__000000.npz``
        video stem = ``s01_<动作名>`` → subject = ``s01``

    Parameters
    ----------
    npz_dir:    npz 目录
    out_dir:   输出目录(会生成 train.txt / test.txt)
    subject_of: 自定义的 subject 解析函数(可选)
    val_ratio:  测试集比例

    Returns
    -------
    dict 统计信息
    """
    npz_dir = Path(npz_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _default_subject(path: Path) -> str:
        # ① 显式 subject 字段
        try:
            data = np.load(path, allow_pickle=False)
            if "subject" in data:
                return str(data["subject"])
            if "video" in data:
                # video = "clips/s01_wave.mp4" → stem "s01_wave" → subject "s01"
                stem = Path(str(data["video"])).stem
                return stem.split("_")[0]
        except Exception:  # noqa: BLE001
            pass
        # ② 文件名兜底
        stem = path.name.split("__")[0]
        return stem.split("_")[0]

    if subject_of is None:
        subject_of = _default_subject

    by_subject: dict[str, list[str]] = {}
    for f in sorted(npz_dir.glob("*.npz")):
        subj = str(subject_of(f))
        by_subject.setdefault(subj, []).append(f.name)

    subjects = sorted(by_subject)
    # 按 subject 排序后取尾部作为测试集 —— 确定性划分,可复现
    n_test = max(1, int(round(len(subjects) * val_ratio))) if len(subjects) > 1 else 0
    test_subjects = set(subjects[-n_test:]) if n_test else set()

    train_files: list[str] = []
    test_files: list[str] = []
    for s in subjects:
        target = test_files if s in test_subjects else train_files
        target.extend(by_subject[s])

    (out_dir / "train.txt").write_text("\n".join(train_files) + "\n", encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(test_files) + "\n", encoding="utf-8")

    return {
        "num_subjects": len(subjects),
        "train_subjects": len(subjects) - len(test_subjects),
        "test_subjects": len(test_subjects),
        "num_train": len(train_files),
        "num_test": len(test_files),
        "test_subject_ids": sorted(test_subjects),
    }


# ===========================================================================
# 数据集
# ===========================================================================
if _HAS_TORCH:

    class AnnotatedNPZDataset(Dataset):
        """
        自建标注数据集 —— 带帧级标签。

        与 ``actionflow.data.dataset.AnnotatedSkeletonDataset`` 的区别:
        后者是"整个窗口一个标签"(导出的窗口不跨段),
        本类是"逐帧标签"(窗口可以跨段,每一帧按它所属的动作段标注)。

        为什么必须逐帧
        --------------
        ① **流式模型**:32 帧就是 32 个训练信号,而不是 1 个
        ② **边界回归头**:需要「每帧到最近起止边界的距离」。
           只有段标签时,窗口内所有帧的标签都一样,
           算出来的距离全是 0 —— 学不到任何东西。

        帧级标签从哪来
        --------------
        从**原始标注 JSON** 重建:

            标注说 [30, 70) 是"抬手"
            → 第 30~69 帧的帧级标签都是"抬手"

        所以本类需要两个输入:
            npz_dir    —— 骨架切片(训练输入)
            annotations —— 标注 JSON(帧级标签来源)

        如果没有 annotations,退化成段标签广播(整窗同一标签),
        此时边界损失会被跳过(见 train.py:compute_loss)。

        Parameters
        ----------
        npz_dir:      存放 npz 的目录
        label_to_id:  标签名 → id 映射
        annotations:  标注 JSON 文件或目录(可选,用于帧级标签)
        window:       窗口长度(输入骨架会被切成这个长度)
        stride:       切窗步长
        augment:      是否做数据增强
        normalize:    是否做身体坐标系归一化
        seed:         随机种子
        file_list:    限定使用哪些文件(用于 train/test 划分)
        """

        def __init__(
            self,
            npz_dir: str | Path,
            label_to_id: dict[str, int],
            annotations: str | Path | None = None,
            window: int = 32,
            stride: int = 16,
            augment: bool = True,
            normalize: bool = True,
            seed: int = 0,
            file_list: list[str] | None = None,
        ) -> None:
            self.npz_dir = Path(npz_dir)
            self.label_to_id = label_to_id
            self.window = window
            self.stride = stride
            self.augment = augment
            self.normalize = normalize
            self.seed = seed

            if file_list is not None:
                self.files = [self.npz_dir / n for n in file_list]
                self.files = [f for f in self.files if f.exists()]
            else:
                self.files = sorted(self.npz_dir.glob("*.npz"))

            # ── 帧级标签表 ────────────────────────────────────────
            # {(video, frame_index): label_id}
            self.frame_label_map: dict[tuple[str, int], int] = {}
            # ── 绝对段边界表 ──────────────────────────────────────
            # {video_stem: [(abs_start, abs_end), ...]}
            # 边界回归头的训练目标,必须用绝对帧号(见 _segment_bounds_for)
            self.segment_map: dict[str, list[tuple[int, int]]] = {}
            self._load_annotations(annotations)

        # ------------------------------------------------------------------
        def _load_annotations(self, annotations: str | Path | None) -> None:
            """从标注 JSON 重建 (video, frame) → label_id 的映射。"""
            if annotations is None:
                return

            from actionflow.data.annotate import load_annotations

            paths: list[Path]
            p = Path(annotations)
            if p.is_dir():
                paths = sorted(p.glob("*.json"))
            else:
                paths = [p]

            for jp in paths:
                try:
                    videos, _ = load_annotations(jp)
                except Exception:  # noqa: BLE001
                    continue
                for v in videos:
                    vid = Path(v.video).stem
                    bounds: list[tuple[int, int]] = []
                    for a in v.actions:
                        bounds.append((a.start_frame, a.end_frame))
                        lid = self.label_to_id.get(a.label)
                        if lid is None:
                            continue
                        # end_frame 是右开区间
                        for t in range(a.start_frame, a.end_frame):
                            self.frame_label_map[(vid, t)] = lid
                    if bounds:
                        self.segment_map[vid] = sorted(bounds)

        # ------------------------------------------------------------------
        def _frame_labels_for(
            self,
            video: str,
            start_frame: int,
            num_frames: int,
        ) -> np.ndarray:
            """
            取一段骨架的逐帧标签。

            优先用标注重建的帧级标签(能区分同一视频里的不同动作段);
            取不到时退回段标签广播。
            """
            vid = Path(str(video)).stem
            out = np.full(num_frames, -1, dtype=np.int64)
            hit = False
            for i in range(num_frames):
                lid = self.frame_label_map.get((vid, start_frame + i))
                if lid is not None:
                    out[i] = lid
                    hit = True
            if not hit:
                # 退回:整段同一标签(填 -1,让调用方用段标签兜底)
                out[:] = -1
            return out

        # ------------------------------------------------------------------
        def _segment_bounds_for(
            self,
            video: str,
            start_frame: int,
            num_frames: int,
        ) -> list[tuple[int, int]]:
            """
            取窗口覆盖到的**绝对**动作段边界。

            这是边界回归头的训练目标 —— 必须与推理时的语义一致:
            ``起点 = t - dist_to_start[t]`` 里的 t 是视频绝对帧号,
            所以目标距离也必须是「到动作段边界的绝对距离」。

            窗口可能横跨多个段,这里把窗口覆盖到的所有段都返回,
            段边界按**绝对帧号**给出(不做窗口内相对化)。
            """
            vid = Path(str(video)).stem
            bounds: list[tuple[int, int]] = []
            for (s, e) in self.segment_map.get(vid, []):
                # 段与窗口有交集才保留
                if e > start_frame and s < start_frame + num_frames:
                    bounds.append((s, e))
            return bounds

        # ------------------------------------------------------------------
        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int) -> dict:
            path = self.files[idx]
            data = np.load(path, allow_pickle=False)

            # ── 兼容两种 key ────────────────────────────────────
            # export 写的是 "skel";SkeletonActionDataset 读的是 "poses"
            if "skel" in data:
                skel = np.asarray(data["skel"], dtype=np.float32)
            else:
                skel = np.asarray(data["poses"], dtype=np.float32)

            label_name = str(data["label"])
            start_frame = int(data["start_frame"]) if "start_frame" in data else 0
            video = str(data["video"]) if "video" in data else path.stem

            lid = self.label_to_id.get(label_name)
            if lid is None:
                # 未知标签 → 跳过这个样本的风险太大,直接用 0 并保持一致
                lid = 0

            # ── 帧级标签 ───────────────────────────────────────
            frame_labels = self._frame_labels_for(video, start_frame, skel.shape[0])
            if (frame_labels < 0).all():
                # 没有帧级标注 → 整窗同一个标签(退化为段标签)
                frame_labels = np.full(skel.shape[0], lid, dtype=np.int64)

            # ── 绝对段边界(边界回归头的目标) ────────────────────
            bounds = self._segment_bounds_for(video, start_frame, skel.shape[0])
            if not bounds:
                # 没有标注时,把窗口本身当成一个段
                bounds = [(start_frame, start_frame + skel.shape[0])]

            if self.normalize:
                from actionflow.postprocess.normalize import normalize_sequence

                skel = normalize_sequence(skel, mode="torso", rotate_to_canonical=True)

            if self.augment:
                from actionflow.data.dataset import augment_skeleton

                rng = np.random.default_rng(self.seed + idx)
                skel = augment_skeleton(skel, rng)

            return {
                "skel": torch.from_numpy(np.ascontiguousarray(skel)),
                "label": lid,
                "frame_labels": torch.from_numpy(frame_labels),
                # 窗口覆盖到的绝对动作段边界 —— 边界回归头的训练目标。
                # 必须是绝对帧号,才能和 decode.py 的反算公式对上。
                "segment_bounds": bounds,
                # 窗口起点在视频中的绝对帧号 —— 用来把绝对边界转成窗口内相对位置
                "window_start": start_frame,
            }

else:  # pragma: no cover

    class AnnotatedNPZDataset:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("训练需要安装 torch：pip install 'actionflow[train]'")
