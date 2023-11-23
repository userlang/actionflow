"""
ActionFlow 命令行入口。

用法:
    # 提取姿态（不需要模型）
    python -m actionflow.cli pose --video demo.mp4 --out poses.npz

    # 提取姿态并可视化
    python -m actionflow.cli pose --video demo.mp4 --visualize out.mp4

    # 端到端推理（需要已训练模型）
    python -m actionflow.cli infer --video demo.mp4 --model weights/af.pth

    # 实时摄像头
    python -m actionflow.cli live --model weights/af.pth

    # 调试解码算法（手工构造概率序列）
    python -m actionflow.cli decode-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from actionflow.core.config import ActionFlowConfig, load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="配置文件路径 (yaml/json)")
    p.add_argument("--onnx", type=str, default=None, help="RTMPose ONNX 权重路径")


def cmd_pose(args: argparse.Namespace) -> int:
    """提取姿态关键点。"""
    from actionflow.infer.pipeline import ActionFlowPipeline
    from actionflow.io.video import open_source
    from actionflow.io.visualize import draw_skeleton

    cfg = ActionFlowConfig()
    if args.config:
        cfg = load_config(args.config)
    if args.onnx:
        cfg.pose.onnx_path = args.onnx

    pipe = ActionFlowPipeline(cfg)
    src = open_source(args.video)
    print(f"视频: {args.video}  fps={src.fps:.1f}  {src.width}x{src.height}")

    import cv2

    poses = []
    dt = 1.0 / max(src.fps, 1e-6)
    writer = None
    frame_idx = 0

    for frame, _ts in src.iter_frames():
        kp = pipe.process_frame(frame, dt=dt)
        poses.append(kp)

        if args.visualize:
            vis = frame.copy()
            # ⚠️ 画骨架必须用**像素坐标**(process_frame 顺手存在 last_raw_pose)。
            # kp 是归一化后的身体坐标(±2 以内),直接画会全挤在左上角。
            draw_skeleton(vis, pipe.last_raw_pose, conf_threshold=cfg.pose.keypoint_thresh)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(
                    args.visualize, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h)
                )
            writer.write(vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  已处理 {frame_idx} 帧...")

    src.release()
    if writer:
        writer.release()
        print(f"可视化视频已保存: {args.visualize}")

    arr = np.stack(poses).astype(np.float32) if poses else np.zeros((0, 17, 3), np.float32)
    out = Path(args.out) if args.out else Path("output") / "poses.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, poses=arr, fps=src.fps)
    print(f"骨架已保存: {out}  shape={arr.shape}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    """端到端推理。"""
    import torch

    from actionflow.infer.pipeline import ActionFlowPipeline
    from actionflow.models.factory import build_model_from_config

    cfg = ActionFlowConfig()
    if args.config:
        cfg = load_config(args.config)
    if args.onnx:
        cfg.pose.onnx_path = args.onnx

    model = build_model_from_config(cfg.model)
    if args.model:
        ckpt = torch.load(args.model, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        print(f"已加载模型: {args.model}")
    else:
        print("[警告] 未指定 --model，使用随机初始化权重（结果无意义，仅验证流程）")

    pipe = ActionFlowPipeline(cfg, model=model)
    segments = pipe.process_video(args.video)

    print(f"\n共 {len(segments)} 个动作段:")
    for seg in segments:
        name = cfg.labels[seg.label_id] if cfg.labels else f"action_{seg.label_id}"
        print(f"  [{seg.start_time:6.2f} - {seg.end_time:6.2f}s] {name:20s} conf={seg.confidence:.3f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.to_dict(cfg.labels or None) for s in segments]
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON 已保存: {out}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """实时摄像头模式。"""
    import torch

    from actionflow.infer.pipeline import ActionFlowPipeline
    from actionflow.io.video import CameraSource
    from actionflow.io.visualize import draw_action_bar, draw_hud, draw_skeleton
    from actionflow.models.factory import build_model_from_config

    cfg = ActionFlowConfig()
    if args.config:
        cfg = load_config(args.config)
    if args.onnx:
        cfg.pose.onnx_path = args.onnx

    model = build_model_from_config(cfg.model)
    if args.model:
        ckpt = torch.load(args.model, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)

    pipe = ActionFlowPipeline(cfg, model=model)
    cam = CameraSource(args.device)
    print(f"摄像头已开启 (device={args.device})，按 'q' 退出")

    import cv2

    dt = 1.0 / max(cam.fps, 1e-6)
    buf = []
    emitted_keys = set()

    for frame, ts in cam.iter_frames():
        kp = pipe.process_frame(frame, dt=dt)
        buf.append(kp)
        if len(buf) > cfg.model.window:
            buf.pop(0)

        vis = frame.copy()
        # 画骨架用像素坐标,理由同 cmd_pose
        draw_skeleton(vis, pipe.last_raw_pose, conf_threshold=cfg.pose.keypoint_thresh)

        if len(buf) == cfg.model.window and model is not None:
            import numpy as np
            import torch

            skel = torch.from_numpy(np.stack(buf)).float().unsqueeze(0)
            model.eval()
            with torch.no_grad():
                out = model.forward_probs(skel)

            # 带边界头的模型返回 (probs, boundary),普通模型只返回 probs
            if isinstance(out, tuple):
                probs_t, boundary_t = out
                boundary = boundary_t[0].cpu().numpy() if boundary_t is not None else None
            else:
                probs_t = out
                boundary = None

            probs = probs_t[0].cpu().numpy()
            segs = pipe.decode(
                probs[:-1],
                fps=cam.fps,
                boundary_dist=None if boundary is None else boundary[:-1],
            )

            for seg in segs:
                key = (round(seg.start_frame, 3), round(seg.end_frame, 3), seg.label_id)
                if key not in emitted_keys:
                    emitted_keys.add(key)
                    name = cfg.labels[seg.label_id] if cfg.labels else f"action_{seg.label_id}"
                    print(f"[{seg.start_time:6.2f}-{seg.end_time:6.2f}s] {name}  conf={seg.confidence:.3f}")

            draw_action_bar(vis, segs, max(ts, 1e-6), ts, cfg.labels or None)

        draw_hud(vis, [f"ActionFlow  t={ts:.1f}s", f"buffer={len(buf)}/{cfg.model.window}"])
        cv2.imshow("ActionFlow", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    return 0


def cmd_decode_test(args: argparse.Namespace) -> int:
    """
    解码算法调试 —— 手工构造概率序列，肉眼验证输出对不对。

    不需要视频、不需要模型、不需要 ONNX。用来单独理解 Layer 3。
    """
    from actionflow.postprocess.decode import (
        build_transition_matrix,
        decode_segments,
        format_segments,
        hysteresis_threshold,
        viterbi_decode,
    )

    C, T, FPS = 4, 120, 30.0
    probs = np.zeros((T, C), dtype=np.float64)

    # 手工设计一段「有噪声的」动作序列：
    #   0.0-1.0s  动作0 (抬手)
    #   1.0-2.0s  动作1 (握拳)
    #   2.0-3.0s  动作2 (抬腿)
    #   3.0-4.0s  动作3 (转身)
    truth = [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30

    rng = np.random.default_rng(0)
    for t, c in enumerate(truth):
        # 基础置信度 0.9，加上噪声
        conf = 0.9 + rng.normal(0, 0.05)
        probs[t, c] = max(0.0, min(1.0, conf))
        # 其余类别分到剩余概率
        rest = (1.0 - probs[t, c]) / (C - 1)
        for k in range(C):
            if k != c:
                probs[t, k] = rest * (0.5 + rng.random())

    # ── 注入三个典型问题 ────────────────────────────────────────
    print("=" * 68)
    print("注入的问题：")
    print("  ① 第 35 帧误判为 动作3  → 朴素 argmax 会把动作1 切成两段")
    print("  ② 第 65 帧置信度骤降到 0.25 → 动作2 会被切断")
    print("  ③ 第 95 帧孤立高置信度 0.85 → 会产生一个 0.03s 的假段")
    print("=" * 68)

    probs[35, :] = 0.05
    probs[35, 3] = 0.90
    probs[65, :] = 0.05
    probs[65, 2] = 0.25
    probs[95, :] = 0.05
    probs[95, 1] = 0.85

    print("\n【1】朴素逐帧 argmax（没有解码层）:")
    naive = probs.argmax(1)
    runs = []
    cur, start = naive[0], 0
    for t in range(1, T):
        if naive[t] != cur:
            runs.append((start, t, int(cur)))
            cur, start = naive[t], t
    runs.append((start, T, int(cur)))
    for s, e, c in runs:
        print(f"    [{s/FPS:5.2f}-{e/FPS:5.2f}s] 动作{c}")
    print(f"    → 共 {len(runs)} 段（碎片很多）")

    print("\n【2】Hysteresis 双阈值 (high=0.7, low=0.4):")
    segs = hysteresis_threshold(probs, high=0.7, low=0.4)
    for s, e, c in segs:
        print(f"    [{s/FPS:5.2f}-{e/FPS:5.2f}s] 动作{c}")
    print(f"    → 共 {len(segs)} 段")

    print("\n【3】Viterbi 动态规划路径（只看类别，不看置信度）:")
    trans = build_transition_matrix(C, stay_prob=0.9, switch_prob=0.05)
    path = viterbi_decode(np.log(np.clip(probs, 1e-12, None)), trans)
    vruns = []
    cur, start = path[0], 0
    for t in range(1, T):
        if path[t] != cur:
            vruns.append((start, t, int(cur)))
            cur, start = path[t], t
    vruns.append((start, T, int(cur)))
    for s, e, c in vruns:
        print(f"    [{s/FPS:5.2f}-{e/FPS:5.2f}s] 动作{c}")
    print(f"    → 共 {len(vruns)} 段（路径连续，误判被拉回）")

    print("\n【4】完整解码（Viterbi + Hysteresis + 最小时长 0.3s）:")
    final = decode_segments(probs, fps=FPS, high=0.7, low=0.4, min_duration=0.3, use_viterbi=True)
    print(format_segments(final))
    print(f"    → 共 {len(final)} 段")

    print("\n" + "=" * 68)
    print("结论：三层解码各治一种病：")
    print("  Hysteresis → 治「段被切断」")
    print("  Viterbi    → 治「孤立误判」")
    print("  最小时长   → 治「0.03s 假阳性段」")
    print("=" * 68)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """打印默认配置。"""
    cfg = ActionFlowConfig()
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="actionflow",
        description="ActionFlow — 流式人体原子动作识别引擎",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pose = sub.add_parser("pose", help="提取姿态关键点")
    p_pose.add_argument("--video", required=True, help="视频文件 / 目录 / 摄像头编号")
    p_pose.add_argument("--out", default=None, help="输出 npz 路径")
    p_pose.add_argument("--visualize", default=None, help="输出可视化视频路径")
    _add_common(p_pose)
    p_pose.set_defaults(func=cmd_pose)

    p_infer = sub.add_parser("infer", help="端到端推理")
    p_infer.add_argument("--video", required=True, help="视频文件 / 目录")
    p_infer.add_argument("--model", default=None, help="训练好的分类器权重 (.pth)")
    p_infer.add_argument("--out", default=None, help="输出 JSON 路径")
    _add_common(p_infer)
    p_infer.set_defaults(func=cmd_infer)

    p_live = sub.add_parser("live", help="实时摄像头模式")
    p_live.add_argument("--device", type=int, default=0, help="摄像头编号")
    p_live.add_argument("--model", default=None, help="训练好的分类器权重 (.pth)")
    _add_common(p_live)
    p_live.set_defaults(func=cmd_live)

    p_dec = sub.add_parser("decode-test", help="调试解码算法（无需任何依赖）")
    p_dec.set_defaults(func=cmd_decode_test)

    p_cfg = sub.add_parser("config", help="打印默认配置")
    p_cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
