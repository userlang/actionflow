"""
训练脚本 —— Layer 2 动作分类器。

用法:
    python -m actionflow.train --data /path/to/nturgbd --epochs 30
    python -m actionflow.train --data /path/to/nturgbd --config configs/ntu.yaml

训练产物:
    weights/actionflow_skel.pth    模型权重
    logs/train.log                训练日志
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from actionflow.core.config import ActionFlowConfig, load_config, save_config
from actionflow.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 ActionFlow 骨架动作分类器")
    p.add_argument("--data", required=True, help="数据集根目录（NTU 或自定义 npz 目录）")
    p.add_argument("--config", default=None, help="配置文件路径")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=60)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default="weights/actionflow_skel.pth")
    p.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    p.add_argument("--amp", action="store_true", help="混合精度训练")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--boundary-weight",
        type=float,
        default=0.1,
        help=(
            "边界回归损失相对分类损失的权重(仅 arch=boundary 生效)。"
            "实测: ≤0.1 时分类正常收敛,≥0.3 时回归项压过交叉熵、"
            "分类准确率卡死在随机水平 —— 故默认取 0.1。"
        ),
    )
    # ── 模型架构 ────────────────────────────────────────────────
    p.add_argument(
        "--arch",
        default=None,
        choices=["offline", "streaming", "boundary", "fusion"],
        help=(
            "模型架构: "
            "offline=整段打一个标签; "
            "streaming=流式逐帧输出(默认); "
            "boundary=流式+精确边界; "
            "fusion=骨架+视频双路"
        ),
    )
    p.add_argument(
        "--causal",
        action="store_true",
        help="给 offline 架构加因果掩码(等价于用 streaming 架构)",
    )
    # ── 数据集类型 ─────────────────────────────────────────────────────
    p.add_argument(
        "--dataset-type",
        default="ntu",
        choices=["ntu", "annotated"],
        help=(
            "ntu=NTU .skeleton 目录(默认); "
            "annotated=自建标注 npz 目录(需要 --annotations)"
        ),
    )
    p.add_argument(
        "--annotations",
        default=None,
        help="标注 JSON 目录(annotated 模式用,提供帧级标签)",
    )
    p.add_argument(
        "--labels-json",
        default=None,
        help="标签映射 labels.json(annotated 模式用;默认 <data>/../labels.json)",
    )
    p.add_argument(
        "--splits",
        default=None,
        help="划分目录,含 train.txt/test.txt(annotated 模式用)",
    )
    return p.parse_args(argv)


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def build_dataloaders(args: argparse.Namespace) -> tuple:
    """
    构建 train / test DataLoader。

    两种数据集类型
    -------------
    ``ntu``
        NTU RGB+D 的 ``.skeleton`` 目录。按 subject id 划分。
    ``annotated``
        自建标注数据集(``export_to_npz_dataset`` 导出的 npz 目录)。
        划分来自 ``splits/train.txt`` / ``splits/test.txt``,
        标签映射来自 ``labels.json``。

        ⚠️ 为什么需要这个开关:
        ``NTUActionDataset`` 扫描的是 ``*.skeleton`` 文件,
        给它一个 npz 目录会建出**空数据集** ——
        训练"跑完了"但样本数是 0,不报错,极具欺骗性。
    """
    from actionflow.data.dataset import collate_batch

    if getattr(args, "dataset_type", "ntu") == "annotated":
        return _build_annotated_loaders(args)

    from actionflow.data.dataset import NTUActionDataset

    train_set = NTUActionDataset(
        args.data,
        split="train",
        num_classes=args.num_classes,
        window=args.window,
        stride=args.stride,
        augment=not args.no_augment,
        seed=args.seed,
    )
    test_set = NTUActionDataset(
        args.data,
        split="test",
        num_classes=args.num_classes,
        window=args.window,
        stride=args.stride,
        augment=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, test_loader


def _build_annotated_loaders(args: argparse.Namespace) -> tuple:
    """自建标注数据集的 DataLoader。"""
    from actionflow.data.annotated import (
        AnnotatedNPZDataset,
        build_label_map,
        labels_in_order,
        load_label_map,
    )
    from actionflow.data.dataset import collate_batch

    data_dir = Path(args.data)

    # ── 标签映射 ────────────────────────────────────────────────────
    labels_json = args.labels_json
    if labels_json is None:
        cand = data_dir.parent / "labels.json"
        labels_json = str(cand) if cand.exists() else None

    if labels_json and Path(labels_json).exists():
        label_to_id = load_label_map(labels_json)
        logger.info(f"标签映射: {labels_json}")
    else:
        # 没有映射文件就现场建一个 —— 否则所有标签都会静默变成 id 0
        label_to_id = build_label_map(data_dir)
        logger.info(f"标签映射已生成 ({len(label_to_id)} 类): {data_dir.parent / 'labels.json'}")

    if not label_to_id:
        raise ValueError(f"{data_dir} 里没有找到任何带 label 的 npz 文件")

    # ── 划分 ────────────────────────────────────────────────────────
    train_files: list[str] | None = None
    test_files: list[str] | None = None

    splits_dir = args.splits
    if splits_dir is None:
        cand = data_dir.parent / "splits"
        splits_dir = str(cand) if cand.exists() else None

    if splits_dir:
        sp = Path(splits_dir)
        tr = sp / "train.txt"
        te = sp / "test.txt"
        if tr.exists():
            train_files = [l.strip() for l in tr.read_text(encoding="utf-8").splitlines() if l.strip()]
        if te.exists():
            test_files = [l.strip() for l in te.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ── 只有 1 个人时的兜底 ──────────────────────────────────────
    # build_split 是按「人」划分的,单人数据集会得到空的 test.txt。
    # 那样 evaluate 永远返回 0%,而且 best_acc 卡在 0 → 权重一次都不存。
    # 所以退回到"按窗口随机留出 20%",并如实打警告:
    # 这个指标衡量不了跨人泛化,只是同一个人的不同片段。
    if test_files is not None and len(test_files) == 0 and train_files:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_files))
        n_test = max(1, int(round(len(train_files) * 0.2)))
        holdout = sorted(train_files[i] for i in idx[:n_test])
        train_files = [f for f in train_files if f not in set(holdout)]
        test_files = holdout
        logger.warning(
            f"⚠️ 测试集为空(数据只有 1 个人,无法按人划分)。"
            f"已退回随机留出 {n_test} 个窗口做测试 —— "
            f"这个准确率只能说明模型记住了你,不代表换个人也认得出。"
        )

    train_set = AnnotatedNPZDataset(
        data_dir,
        label_to_id,
        annotations=args.annotations,
        window=args.window,
        stride=args.stride,
        augment=not args.no_augment,
        seed=args.seed,
        file_list=train_files,
    )
    test_set = AnnotatedNPZDataset(
        data_dir,
        label_to_id,
        annotations=args.annotations,
        window=args.window,
        stride=args.stride,
        augment=False,
        seed=args.seed,
        file_list=test_files,
    )

    # 把标签名按 id 顺序挂到 args 上,训练结束会存进 checkpoint
    args.label_list = labels_in_order(label_to_id)
    args.num_classes = len(label_to_id)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=len(train_set) > args.batch_size,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> dict:
    """评估模型：返回 accuracy 和 per-class 准确率。"""
    model.eval()
    correct = 0
    total = 0
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)

    for batch in loader:
        skel = batch["skel"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        frame_labels = batch.get("frame_labels")
        if frame_labels is not None:
            frame_labels = frame_labels.to(device, non_blocking=True)

        with torch.no_grad():
            out = model(skel)
            logits = out[0] if isinstance(out, tuple) else out

            if logits.dim() == 3:
                # 逐帧输出 (B, T, C)
                pred = logits.argmax(dim=-1)              # (B, T)
                if frame_labels is not None:
                    # 有帧级标签:逐帧比,统计的是"帧级准确率"
                    pred = pred.reshape(-1)
                    tgt = frame_labels.reshape(-1)
                else:
                    # 只有段标签:取窗口最后一帧的预测来比
                    pred = pred[:, -1]
                    tgt = label
            else:
                # 整段输出 (B, C)
                pred = logits.argmax(dim=-1)
                tgt = label

        correct += (pred == tgt).sum().item()
        total += tgt.numel()

        for c in tgt.unique():
            mask = tgt == c
            per_class_correct[c] += (pred[mask] == c).sum().item()
            per_class_total[c] += mask.sum().item()

    return {
        "accuracy": correct / max(total, 1),
        "per_class_accuracy": (per_class_correct.float() / per_class_total.clamp(min=1)).tolist(),
        "total": total,
    }


def compute_loss(
    model,
    skel,
    label,
    criterion,
    device,
    frame_labels=None,
    boundary_weight=None,
    segment_bounds=None,
    window_start=None,
):
    """
    按架构计算训练损失。

    为什么需要这个函数
    ------------------
    不同架构的输出形状不一样:

        offline  : forward → (B, C)              标签是 (B,)
        streaming: forward → (B, T, C)           标签是 (B,)  ← 逐帧输出怎么和帧标签对?
        boundary : forward → (B, T, C), (B, T, 2)

    逐帧模型的训练有个**天然的矛盾**:
        一个窗口 (T=32 帧) 对应的段标签只有一个(整段是"抬手"),
        但模型输出 32 个逐帧结果。

    两种解法
    ---------
        ① **帧级标签**（推荐）
           数据集提供每一帧的标签,做逐帧交叉熵。
           监督信号最充分 —— 32 帧就是 32 个训练信号。
           本项目的 ``AnnotatedSkeletonDataset`` 就提供这个。

        ② **段级标签**（退而求其次）
           只有段标签时,把段标签广播到每一帧。
           监督信号少,而且边界附近的帧其实标错了
           (动作正在切换时,前后几帧其实是模糊的)。

    边界头需要的是**帧级标签**才能学:
        只有段标签的话,窗口内所有帧的标签都一样,
        算出来的"到最近边界的距离"全是 0 —— 学不到任何东西。
        所以没给帧标签时,边界损失直接跳过(而不是喂垃圾数据)。

    Parameters
    ----------
    frame_labels : (B, T) | None
        每一帧的标签。给了就用它,没给就退化成段标签广播。
    """
    out = model(skel)

    if isinstance(out, tuple):
        logits, boundary = out
    else:
        logits, boundary = out, None

    if logits.dim() == 2:
        # 整段一个结果
        loss = criterion(logits, label)
    else:
        # 逐帧结果
        B, T, C = logits.shape
        if frame_labels is not None:
            # ① 有帧级标签 —— 最好的情况
            loss = criterion(logits.reshape(-1, C), frame_labels.reshape(-1))
        else:
            # ② 只有段标签 —— 广播到每一帧
            #    ⚠️ 注意用新变量,不要覆盖 frame_labels ——
            #    下面判断"要不要加边界损失"时还要靠它是不是 None。
            seg_labels = label.unsqueeze(1).expand(B, T)
            loss = criterion(logits.reshape(-1, C), seg_labels.reshape(-1))

    # ── 边界回归损失 ────────────────────────────────────────────
    # ⚠️ 只有在**真的有帧级标签**时才加。
    #    没有帧标签时,窗口内每一帧的标签都一样,
    #    算出来的"到最近边界的距离"全是 0 —— 学不到任何东西,
    #    还会用垃圾目标污染梯度。所以直接跳过。
    if boundary is not None and frame_labels is not None:
        from actionflow.models.boundary import (
            boundary_regression_loss,
            compute_boundary_targets,
        )

        # 构造真值:每帧到最近起止边界的距离。
        #
        # ⚠️ 必须用**绝对段边界**算,不能用窗口内的标签变化推断。
        #    推理时 decode.py 的反算公式是
        #        起点 = t - dist_to_start[t]
        #    这里的 t 是视频绝对帧号,所以训练目标也必须是绝对距离。
        #    详见 boundary.py::compute_boundary_targets 的说明。
        dist_start, dist_end = compute_boundary_targets(
            frame_labels,
            num_classes=logits.shape[-1],
            segment_bounds=segment_bounds,
            window_start=window_start,
        )
        target = torch.stack([dist_start, dist_end], dim=-1).to(device)

        # 有效帧:标签有效(>= 0)的帧才参与回归
        valid = frame_labels >= 0
        if valid.any():
            # ⚠️ boundary_weight 控制边界项与分类项的权衡。
            #    回归损失已经归一化到 O(1) 量级(见 boundary.py 的说明),
            #    默认 1.0 表示两者等权。
            #    调大 → 更看重时间戳精度;调小 → 更看重分类准确率。
            w = boundary_weight if boundary_weight is not None else 1.0
            loss = loss + w * boundary_regression_loss(boundary, target, valid)

    return loss


def train(args: argparse.Namespace) -> int:
    from actionflow.models.factory import build_model_from_config

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None  # type: ignore

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_logging()

    device = torch.device(args.device)
    logger.info(f"设备: {device}")

    # ── 配置 ──────────────────────────────────────────────────────
    cfg = ActionFlowConfig()
    if args.config:
        cfg = load_config(args.config)
    cfg.model.num_classes = args.num_classes
    cfg.model.window = args.window
    if getattr(args, "arch", None):
        cfg.model.arch = args.arch
    if getattr(args, "causal", None):
        cfg.model.causal = args.causal
        cfg.model.arch = "offline"

    logger.info(f"模型架构: {cfg.model.arch}")

    # ── 数据 ──────────────────────────────────────────────────────
    train_loader, test_loader = build_dataloaders(args)
    logger.info(f"训练样本: {len(train_loader.dataset)}  测试样本: {len(test_loader.dataset)}")

    # 标签名表 —— 推理时把 label_id 转成可读名字。
    # 必须放在 build_dataloaders **之后**:annotated 模式会在那里
    # 根据实际数据统计出类别数并生成标签表,提前取会拿到空值。
    label_list = getattr(args, "label_list", None)
    if label_list:
        cfg.labels = list(label_list)
        cfg.model.num_classes = len(cfg.labels)
        logger.info(f"类别数: {cfg.model.num_classes}  标签: {cfg.labels}")

    # ⚠️ 空数据集检查 —— NTU 目录写错时不会报错,只会训练 0 个样本。
    if len(train_loader.dataset) == 0:
        raise ValueError(
            f"训练集为空！检查 --data 目录: {args.data}\n"
            f"  ntu 模式: 目录里应该有 *.skeleton 文件\n"
            f"  annotated 模式: 目录里应该有 export 出来的 *.npz 文件"
        )

    # ── 模型 ──────────────────────────────────────────────────────
    model = build_model_from_config(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {n_params/1e6:.3f}M")

    # ── 优化器 ────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir="logs/tensorboard") if SummaryWriter else None

    best_acc = 0.0
    history = []

    # ── 训练循环 ──────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            skel = batch["skel"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            # 帧级标签(可选)—— 逐帧模型和边界头需要它
            frame_labels = batch.get("frame_labels")
            if frame_labels is not None:
                frame_labels = frame_labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # 不同架构的 forward 输出形状不同,这里统一处理。
            #   offline  : (B, C)        整段一个标签
            #   streaming: (B, T, C)     每帧一个标签
            #   boundary : (B, T, C) + (B, T, 2)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                loss = compute_loss(
                    model, skel, label, criterion, device, frame_labels,
                    boundary_weight=args.boundary_weight,
                    segment_bounds=batch.get("segment_bounds"),
                    window_start=batch.get("window_start"),
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            running_loss += loss.item()
            n_batches += 1

            if step % 50 == 0:
                logger.info(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={running_loss/max(n_batches,1):.4f}"
                )

        sched.step()
        train_loss = running_loss / max(n_batches, 1)
        metrics = evaluate(model, test_loader, device, args.num_classes)
        elapsed = time.time() - t0

        logger.info(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | acc {metrics['accuracy']*100:.2f}% "
            f"| {elapsed:.1f}s"
        )
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("acc/test", metrics["accuracy"], epoch)

        history.append({"epoch": epoch, "loss": train_loss, "acc": metrics["accuracy"]})

        # ── 保存最优模型 ──────────────────────────────────────────
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "accuracy": best_acc,
                    "config": cfg.to_dict(),
                    "labels": cfg.labels,
                },
                out_path,
            )
            logger.info(f"  ✓ 保存最优模型 (acc={best_acc*100:.2f}%) → {out_path}")

    if writer is not None:
        writer.close()

    # ── 保存训练历史 + 配置 ───────────────────────────────────────
    hist_path = out_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_config(cfg, out_path.with_suffix(".config.json"))

    logger.info(f"\n训练完成。最佳准确率: {best_acc*100:.2f}%")
    logger.info(f"模型: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
