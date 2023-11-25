"""
Evaluate — 动作段级别的评估
============================

为什么需要这个工具
------------------
``train.py`` 里的 ``evaluate()`` 只看**帧级准确率**:
「这一帧的分类对不对」。

但 ActionFlow 的产出是**动作段** —— 带起止时间戳的一段。
帧级准确率高 ≠ 段级结果好。反例:
    预测段的边界整体偏了 20 帧 → 帧级准确率仍然可能不低,
    但作为「带时间戳的动作序列」这个产出是错的。

所以必须有一个**段级**的评估工具,回答三个问题:

    ① 段**找对了吗**?      → 段级 Precision / Recall / F1
    ② 段**边界准吗**?      → 平均 IoU、边界误差(帧)
    ③ 分类**对吗**?        → 类别准确率

匹配规则
--------
一个预测段和一个真值段能配成一对,当且仅当:

    ① 类别相同
    ② 两者都有重叠(IoU > 0)

配对用**贪心最大 IoU** —— 一段视频里同一个真值段不会被两个预测段
重复占用。这一条很重要:否则「模型把一个动作切成了三段」会被算成
「3 个命中」,指标虚高。

用法
----
    # 评估训练好的模型
    python -m actionflow.evaluate \\
        --model weights/my_action.pth \\
        --annotations data/annotations \\
        --npz-dir data/npz \\
        --videos data/clips

    # 只看某几个视频
    python -m actionflow.evaluate --model weights/my_action.pth \\
        --annotations data/annotations --npz-dir data/npz \\
        --videos data/clips/s03_a.mp4 data/clips/s03_b.mp4

    # 用真值帧级标签做「上界」参考:直接把标注当成预测段
    python -m actionflow.evaluate --annotations data/annotations \\
        --npz-dir data/npz --oracle
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class Segment:
    """一个动作段(帧号,右开区间)。"""

    label: str
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


@dataclass
class VideoResult:
    """一个视频的评估结果。"""

    video: str
    n_gt: int = 0
    n_pred: int = 0
    n_matched: int = 0
    ious: list[float] = field(default_factory=list)
    boundary_errors: list[float] = field(default_factory=list)
    label_correct: int = 0  # 类别判对的段数(不管边界)

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gt if self.n_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.ious)) if self.ious else 0.0

    @property
    def mean_boundary_error(self) -> float:
        return float(np.mean(self.boundary_errors)) if self.boundary_errors else 0.0

    @property
    def label_accuracy(self) -> float:
        """类别准确率:真值段里有多少个的**类别**被预测出来了(不管边界)。"""
        return self.label_correct / self.n_gt if self.n_gt else 0.0


# ===========================================================================
# IoU 与配对
# ===========================================================================
def temporal_iou(a: Segment, b: Segment) -> float:
    """两个时间段的 IoU。"""
    inter = min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame)
    if inter <= 0:
        return 0.0
    union = max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame)
    return inter / union if union > 0 else 0.0


def match_segments(
    gt: list[Segment],
    pred: list[Segment],
    iou_threshold: float = 0.1,
) -> list[tuple[int, int, float]]:
    """
    把预测段和真值段配对。

    规则
    ----
    ① 类别必须相同
    ② IoU 必须 > ``iou_threshold``
    ③ 贪心最大 IoU:每次配对 IoU 最高的一对,配过的段不能再用

    ③ 保证「模型把一个动作切成三段」不会被算成 3 个命中 ——
    第二个预测段已经没有可配对的真值段了,只能算误检。

    Returns
    -------
    list[(gt_idx, pred_idx, iou)]
    """
    pairs: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    # 所有候选对,按 IoU 降序
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.label != p.label:
                continue
            iou = temporal_iou(g, p)
            if iou > iou_threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi, iou))

    return pairs


# ===========================================================================
# 评估单个视频
# ===========================================================================
def evaluate_video(
    gt: list[Segment],
    pred: list[Segment],
    video: str = "",
    iou_threshold: float = 0.1,
) -> VideoResult:
    """评估一个视频。"""
    res = VideoResult(video=video, n_gt=len(gt), n_pred=len(pred))

    pairs = match_segments(gt, pred, iou_threshold)

    for gi, pi, iou in pairs:
        res.n_matched += 1
        res.ious.append(iou)
        g, p = gt[gi], pred[pi]
        # 边界误差:起点误差 + 终点误差,取平均(帧)
        res.boundary_errors.append(
            (abs(p.start_frame - g.start_frame) + abs(p.end_frame - g.end_frame)) / 2
        )

    # 类别准确率:不管边界,只看标签对不对 —— 「有没有认对这个动作」
    pred_labels = [s.label for s in pred]
    for g in gt:
        if g.label in pred_labels:
            res.label_correct += 1

    return res


# ===========================================================================
# 从标注 / 推理结果构造段
# ===========================================================================
def load_gt_segments(annotations_dir: str | Path) -> dict[str, list[Segment]]:
    """
    从标注目录读出每个视频的真值段。

    视频的 key 用**视频文件名的 stem**(不含扩展名),与预测侧保持一致。
    """
    from actionflow.data.annotate import load_annotations

    out: dict[str, list[Segment]] = {}
    for jp in sorted(Path(annotations_dir).glob("*.json")):
        videos, _ = load_annotations(jp)
        for v in videos:
            key = Path(str(v.video)).stem
            out[key] = [
                Segment(label=a.label, start_frame=a.start_frame, end_frame=a.end_frame)
                for a in v.actions
            ]
    return out


def predict_segments(
    video: str | Path,
    model_path: str | Path,
    npz_path: str | Path | None = None,
    device: str = "cpu",
) -> list[Segment]:
    """
    对一个视频跑完整推理,返回预测段。

    ``npz_path`` 给定时直接用缓存的关键点(快得多,也保证可复现);
    没给就从视频实时抽。
    """
    import torch

    from actionflow.core.config import ActionFlowConfig, load_config
    from actionflow.models.factory import build_model_from_config
    from actionflow.postprocess.decode import decode_segments

    # 配置:优先用权重旁的 sidecar
    side = Path(model_path).with_suffix(".config.json")
    cfg = load_config(side) if side.exists() else ActionFlowConfig()

    model = build_model_from_config(cfg.model)
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    # ── 拿骨架 ──────────────────────────────────────────────────────
    if npz_path and Path(npz_path).exists():
        poses = np.load(npz_path)["keypoints"].astype(np.float32)
        fps = 30.0
    else:
        from actionflow.infer.pipeline import ActionFlowPipeline

        pipe = ActionFlowPipeline(cfg, model=model)
        poses = pipe.extract_poses(str(video))
        fps = 30.0

    if len(poses) == 0:
        return []

    # ── 逐帧分类(和推理管线完全一致的滑窗逻辑)────────────────────
    T_total = poses.shape[0]
    win = int(cfg.model.window)
    stride = max(win // 2, 1)
    starts = list(range(0, max(T_total - win, 0) + 1, stride)) or [0]

    probs_sum = None
    count = np.zeros(T_total, dtype=np.float64)

    with torch.no_grad():
        for st in starts:
            chunk = poses[st:st + win]
            if chunk.shape[0] < win:
                chunk = np.concatenate(
                    [chunk, np.repeat(chunk[-1:], win - chunk.shape[0], axis=0)], axis=0
                )
            skel = torch.from_numpy(np.ascontiguousarray(chunk)).float().unsqueeze(0)
            out = model.forward_probs(skel)
            probs = out[0] if isinstance(out, tuple) else out
            probs = probs[0].cpu().numpy()

            end = min(st + win, T_total)
            if probs_sum is None:
                probs_sum = np.zeros((T_total, probs.shape[-1]), dtype=np.float64)
            probs_sum[st:end] += probs[: end - st]
            count[st:end] += 1

    if probs_sum is None:
        return []
    probs_sum /= np.maximum(count[:, None], 1)

    # ── 解码 ────────────────────────────────────────────────────────
    d = cfg.decode
    segs = decode_segments(
        probs_sum,
        fps=fps,
        high=d.high_threshold,
        low=d.low_threshold,
        min_duration=d.min_duration,
        use_viterbi=d.use_viterbi,
        stay_prob=d.stay_prob,
        switch_prob=d.switch_prob,
        merge_gap=d.merge_gap,
    )

    labels = cfg.labels or []
    out_segments = []
    for s in segs:
        name = labels[s.label_id] if labels and s.label_id < len(labels) else f"action_{s.label_id}"
        out_segments.append(Segment(name, s.start_frame, s.end_frame))
    return out_segments


# ===========================================================================
# 主流程
# ===========================================================================
def aggregate(results: list[VideoResult]) -> dict:
    """把多个视频的结果汇总成一份报告。"""
    n_gt = sum(r.n_gt for r in results)
    n_pred = sum(r.n_pred for r in results)
    n_matched = sum(r.n_matched for r in results)
    label_correct = sum(r.label_correct for r in results)

    p = n_matched / n_pred if n_pred else 0.0
    r = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    all_ious = [v for res in results for v in res.ious]
    all_bnd = [v for res in results for v in res.boundary_errors]

    return {
        "num_videos": len(results),
        "num_gt_segments": n_gt,
        "num_pred_segments": n_pred,
        "num_matched": n_matched,
        "precision": p,
        "recall": r,
        "f1": f1,
        "label_accuracy": label_correct / n_gt if n_gt else 0.0,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_boundary_error_frames": float(np.mean(all_bnd)) if all_bnd else 0.0,
        "median_boundary_error_frames": float(np.median(all_bnd)) if all_bnd else 0.0,
    }


def _bar(value: float, width: int = 30) -> str:
    """进度条。"""
    n = int(round(value * width))
    return "█" * n + "░" * (width - n)


def print_report(results: list[VideoResult], report: dict) -> None:
    """打印人可读的评估报告。"""
    print("=" * 72)
    print("段级评估报告")
    print("=" * 72)

    print(f"\n视频数: {report['num_videos']}")
    print(
        f"真值段: {report['num_gt_segments']}   "
        f"预测段: {report['num_pred_segments']}   "
        f"命中: {report['num_matched']}"
    )

    print("\n── 核心指标 ──")
    for name, key in [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("类别准确率", "label_accuracy"),
    ]:
        v = report[key]
        print(f"  {name:12s} {v * 100:6.2f}%  {_bar(v)}")

    print("\n── 时间戳精度 ──")
    print(f"  平均 IoU           {report['mean_iou']:.4f}")
    print(f"  平均边界误差       {report['mean_boundary_error_frames']:.2f} 帧")
    print(f"  边界误差中位数     {report['median_boundary_error_frames']:.2f} 帧")
    print(f"  (30fps 下 {report['mean_boundary_error_frames'] / 30 * 1000:.0f} ms)")

    print("\n── 逐视频 ──")
    for res in results:
        print(
            f"  {res.video:24s} GT={res.n_gt:3d} PRED={res.n_pred:3d} "
            f"命中={res.n_matched:3d}  F1={res.f1 * 100:6.2f}%  "
            f"IoU={res.mean_iou:.3f}  边界误差={res.mean_boundary_error:5.1f}帧"
        )

    print("\n" + "=" * 72)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ActionFlow 动作段级评估")
    p.add_argument("--model", default=None, help="训练好的权重 (.pth)")
    p.add_argument("--annotations", required=True, help="标注 JSON 目录")
    p.add_argument("--npz-dir", default=None, help="关键点 npz 目录(加速评估)")
    p.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="要评估的视频(文件或目录);默认评估标注目录里提到的所有视频",
    )
    p.add_argument("--iou-threshold", type=float, default=0.1, help="段匹配的 IoU 下限")
    p.add_argument("--out", default=None, help="把报告存成 JSON")
    p.add_argument(
        "--oracle",
        action="store_true",
        help="不跑模型,直接把真值当成预测 —— 看匹配规则本身的上界",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gt_map = load_gt_segments(args.annotations)
    if not gt_map:
        print(f"标注目录里没有可用的标注: {args.annotations}")
        return 1

    # ── 确定要评估哪些视频 ──────────────────────────────────────────
    video_paths: list[Path] = []
    if args.videos:
        for v in args.videos:
            p = Path(v)
            if p.is_dir():
                video_paths.extend(sorted(p.glob("*.mp4")))
            else:
                video_paths.append(p)
    else:
        # 从标注里的 video 字段推断路径: data/annotations/x.json
        # 里的 "data/clips/s01_a.mp4" → 相对于当前工作目录
        for key in gt_map:
            cand = Path("data/clips") / f"{key}.mp4"
            if cand.exists():
                video_paths.append(cand)

    results: list[VideoResult] = []

    for vp in video_paths:
        key = vp.stem
        gt = gt_map.get(key)
        if gt is None:
            print(f"跳过 {key}: 标注目录里没有这个视频")
            continue

        if args.oracle:
            pred = list(gt)
        else:
            if not args.model:
                print("需要 --model(或用 --oracle 只测匹配规则)")
                return 1
            npz = None
            if args.npz_dir:
                cand = Path(args.npz_dir) / f"{key}.npz"
                if cand.exists():
                    npz = cand
            pred = predict_segments(vp, args.model, npz)

        res = evaluate_video(gt, pred, video=key, iou_threshold=args.iou_threshold)
        results.append(res)

    if not results:
        print("没有可评估的视频")
        return 1

    report = aggregate(results)
    print_report(results, report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "report": report,
            "per_video": [
                {
                    "video": r.video,
                    "n_gt": r.n_gt,
                    "n_pred": r.n_pred,
                    "n_matched": r.n_matched,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "mean_iou": r.mean_iou,
                    "mean_boundary_error_frames": r.mean_boundary_error,
                }
                for r in results
            ],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON 报告已保存: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
