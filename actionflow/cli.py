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


def _cfg_for_model(args: argparse.Namespace) -> ActionFlowConfig:
    """
    按 --model 权重自动挑选配置。

    train.py 每次训练都会在权重旁存一份 ``<权重同名>.config.json``
    (里面记着当时生效的 num_classes / labels / window / decode 阈值)。
    建分类器前必须先按这份配置建,否则用 ActionFlowConfig() 默认的
    num_classes=60 去 load 一个 7 类的 checkpoint,load_state_dict
    直接 size mismatch 崩掉。

    优先级:显式 --config > 权重同名 sidecar > 内置默认。
    """
    if getattr(args, "config", None):
        return load_config(args.config)
    model = getattr(args, "model", None)
    if model:
        side = Path(model).with_suffix(".config.json")
        if side.exists():
            print(f"已加载配置: {side}")
            return load_config(side)
    return ActionFlowConfig()


def cmd_infer(args: argparse.Namespace) -> int:
    """端到端推理。"""
    import torch

    from actionflow.infer.pipeline import ActionFlowPipeline
    from actionflow.models.factory import build_model_from_config

    cfg = _cfg_for_model(args)
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

    if getattr(args, "show", False):
        return _infer_show(pipe, cfg, args)

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


def _infer_show(pipe, cfg, args) -> int:
    """
    边播视频边出检测的可视化模式。

    流式语义(和摄像头 live 一致,只是数据源是文件):
      - 维护 32 帧滚动缓冲,每帧对窗口做一次前向,
        取**最后一帧**的概率作为"当前帧预测" —— 因果模型有
        前缀不变性,这和离线整段滑窗是同一套数学。
      - 把每帧的 (C,) 概率拼成全局 (T_global, C) 概率流,
        每攒 16 帧对整条流重跑一次 decode_segments(Hysteresis+
        Viterbi+min_duration),得到**全局时间轴**上的动作段,
        新确认的段实时打印。
      - 画面:骨架 + 左上当前动作(中文) + 底部动作时间轴。

    按键:q 退出。--save 存带框 mp4;--no-window 只存不弹窗(无显示环境用)。
    """
    import time
    from collections import deque

    import cv2
    import torch

    from actionflow.io.video import open_source
    from actionflow.io.visualize import (
        COLORS,
        draw_action_bar,
        draw_skeleton,
        draw_text,
    )

    src = open_source(args.video)
    fps = src.fps
    win = cfg.model.window
    total_frames = src.frame_count or 0
    total_duration = (total_frames / fps) if (total_frames and fps) else 0.0
    pipe.model.eval()

    show_window = not getattr(args, "no_window", False)
    writer = None
    save = getattr(args, "save", None)
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (src.width, src.height)
        )

    print(f"可视化: {args.video}  {src.width}x{src.height} fps={fps:.1f}  "
          f"(按 q 退出{' ; 仅存盘不弹窗' if not show_window else ''})")

    buf: deque = deque(maxlen=win)
    dt = 1.0 / max(fps, 1e-6)
    stream_probs: list[np.ndarray] = []   # 每帧一个 (C,) —— 全局概率流
    all_segs: list = []                    # 最近一次整条流解码的结果
    seen_seg_keys: set = set()             # 已打印过的段,防重复刷屏

    frame_idx = 0
    t_start = time.perf_counter()

    for frame, _ts in src.iter_frames():
        kp = pipe.process_frame(frame, dt=dt)  # 副作用: last_raw_pose = 像素坐标
        buf.append(kp)
        cur_time = frame_idx / fps if fps else frame_idx * dt

        if len(buf) >= 2:
            # 窗口 = 缓冲内全部帧 + 尾部补当前帧到 win。
            # ⚠️ 缓冲不足一窗时(开头 31 帧)必须**尾补**并取当前帧所在
            #    位置的概率 —— 和训练侧 windowize 的补齐完全一致。
            #    若等攒满一窗才开始记录,概率流会比真实时间轴早 win-1 帧,
            #    解码出的段时间戳整体提前 ~1s,画面标注和播放头对不上。
            arr = np.stack(buf)
            n = len(arr)
            if n < win:
                arr = np.concatenate([arr, np.repeat(arr[-1:], win - n, axis=0)], axis=0)
            with torch.no_grad():
                skel = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)
                out = pipe.model.forward_probs(skel)
                probs = out[0] if isinstance(out, tuple) else out
            frame_prob = probs[0, n - 1].cpu().numpy()  # 当前帧的因果预测
            stream_probs.append(frame_prob)

            # 每 16 帧对整条流重解码一次 —— DP 是 numpy,便宜
            if len(stream_probs) % max(win // 2, 1) == 0:
                all_segs = pipe.decode(np.stack(stream_probs), fps=fps)
                for s in all_segs:
                    key = (round(s.start_time, 1), s.label_id)
                    if key not in seen_seg_keys:
                        seen_seg_keys.add(key)
                        name = cfg.labels[s.label_id] if cfg.labels else str(s.label_id)
                        print(f"  ✓ 检出 [{s.start_time:6.2f}-{s.end_time:6.2f}s] {name}  "
                              f"conf={s.confidence:.2f}")

            # 当前帧动作,三级兜底:
            #   ① 已闭合的解码段覆盖当前时刻 → 用段标签(最终答案)
            #   ② 段还在进行中(hysteresis 只在段闭合后才入列) →
            #      用最近 8 帧概率的均值平滑,过 high 阈值才显示,
            #      避免休息期的瞬时噪声(如误报右踢腿)直接上屏
            #   ③ 都不满足 → 显示"待机/过渡"
            covering = [s for s in all_segs if s.start_time <= cur_time <= s.end_time]
            if covering:
                seg = max(covering, key=lambda a: a.confidence)
                cur_label, cur_conf = seg.label_id, seg.confidence
            else:
                tail = np.stack(stream_probs[-8:]).mean(axis=0)
                cur_label = int(tail.argmax())
                cur_conf = float(tail[cur_label])
                if cur_conf < cfg.decode.high_threshold:
                    cur_label = -1
        else:
            cur_label, cur_conf = -1, 0.0

        vis = frame.copy()
        draw_skeleton(vis, pipe.last_raw_pose, conf_threshold=cfg.pose.keypoint_thresh)

        if cur_label >= 0 and cur_conf >= cfg.decode.low_threshold:
            name = cfg.labels[cur_label] if cfg.labels else str(cur_label)
            cv2.rectangle(vis, (8, 8), (min(420, vis.shape[1] - 8), 66), (0, 0, 0), -1)
            cv2.rectangle(vis, (8, 8), (min(420, vis.shape[1] - 8), 66), (0, 0, 255), 2)
            draw_text(vis, f"当前: {name}", (16, 38), font_size=26, color_bgr=(0, 255, 255))
            draw_text(vis, f"conf={cur_conf:.2f}  t={cur_time:.1f}s",
                      (16, 62), font_size=16, color_bgr=(255, 255, 255))
        else:
            cv2.rectangle(vis, (8, 8), (min(300, vis.shape[1] - 8), 44), (0, 0, 0), -1)
            draw_text(vis, f"待机/过渡  t={cur_time:.1f}s",
                      (16, 38), font_size=20, color_bgr=COLORS["text"])

        if all_segs and total_duration > 0:
            draw_action_bar(vis, all_segs, total_duration, cur_time, labels=cfg.labels)
        if writer:
            writer.write(vis)

        if show_window:
            cv2.imshow("ActionFlow 检测", vis)
            # 节流:目标 帧号/fps = 墙钟时间。落后→wait=1 全速追;
            # 提前→最多睡一帧间隔,别把追赶预算一次睡掉。
            behind_ms = int((frame_idx + 1) / fps * 1000 - (time.perf_counter() - t_start) * 1000)
            wait_ms = 1 if behind_ms <= 0 else min(behind_ms, int(1000 / fps))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
        frame_idx += 1

    # 收尾:整条流最后解码一次
    if stream_probs:
        all_segs = pipe.decode(np.stack(stream_probs), fps=fps)
    src.release()
    if writer:
        writer.release()
        print(f"可视化视频已保存: {save}")
    if show_window:
        cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    speed = frame_idx / wall if wall > 0 else 0
    print(f"\n共处理 {frame_idx} 帧, 最终 {len(all_segs)} 段, "
          f"实际速度 {speed:.1f} fps (视频原速 {fps:.1f})")
    return 0




def cmd_live(args: argparse.Namespace) -> int:
    """实时摄像头模式。"""
    import torch

    from actionflow.infer.pipeline import ActionFlowPipeline
    from actionflow.io.video import CameraSource
    from actionflow.io.visualize import draw_action_bar, draw_hud, draw_skeleton
    from actionflow.models.factory import build_model_from_config

    cfg = _cfg_for_model(args)
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
    p_infer.add_argument("--show", action="store_true",
                         help="可视化模式:弹窗播放视频,实时叠加骨架+当前动作")
    p_infer.add_argument("--save", default=None, metavar="OUT.mp4",
                         help="配合 --show:把带标注的画面另存为 mp4")
    p_infer.add_argument("--no-window", action="store_true",
                         help="配合 --show:不开显示窗口(无屏幕环境/纯存盘)")
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
