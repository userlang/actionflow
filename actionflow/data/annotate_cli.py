"""
标注工具的命令行入口。

用法:
    python -m actionflow.data.annotate_cli extract   --video s01_b.mp4 --out a.npz
    python -m actionflow.data.annotate_cli gui       --video s01_b.mp4 --npz a.npz --out a.json
    python -m actionflow.data.annotate_cli cli       --video s01_b.mp4 --npz a.npz --out a.json
    python -m actionflow.data.annotate_cli validate --annotations dir/ --npz-dir dir/
    python -m actionflow.data.annotate_cli export   --annotations dir/ --npz-dir dir/ --out out/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_extract(args: argparse.Namespace) -> int:
    """从视频自动抽关键点,生成待标注的 npz。"""
    from actionflow.data.annotate import extract_keypoints

    print(f"正在从 {args.video} 抽取关键点...")
    kps = extract_keypoints(
        args.video,
        out_npz=args.out,
        onnx_path=args.onnx,
        conf_threshold=args.threshold,
    )
    print(f"✓ 已保存: {args.out}   shape={kps.shape}")

    # 顺便报一下置信度统计,太低说明姿态模型没跑好
    conf = kps[..., 2]
    print(f"  置信度: 均值 {conf.mean():.3f}, 低于 {args.threshold} 的点占 "
          f"{(conf < args.threshold).mean() * 100:.1f}%")
    if (conf < args.threshold).mean() > 0.3:
        print("  ⚠️ 漏检比例偏高,建议人工修正关键点或检查姿态模型")
    return 0


def cmd_cli(args: argparse.Namespace) -> int:
    """纯命令行标注(无 GUI 时的备选方案)。"""
    from actionflow.data.annotate import (
        ActionAnnotation,
        VideoAnnotation,
        load_annotations,
        save_annotations,
    )

    video = str(args.video)
    out_path = Path(args.out)

    # 已存在就续标
    if out_path.exists():
        videos, _ = load_annotations(out_path)
        existing = next((v for v in videos if v.video == video), None)
    else:
        existing = None

    if existing is None:
        from actionflow.io.video import open_source

        src = open_source(args.video)
        num_frames = 0
        for _ in src.iter_frames():
            num_frames += 1
        src.release()
        existing = VideoAnnotation(video=video, fps=src.fps, num_frames=num_frames)
        existing.keypoints = Path(args.npz).name if args.npz else None
        print(f"新标注: {video}  ({num_frames} 帧, {src.fps:.1f} fps)")

    print("\n逐条输入动作段(空行结束):")
    print("  格式: <标签> <起始帧> <结束帧>")
    print("  例:   抬右手 10 55\n")

    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 3:
            print("  格式不对,应该是: 标签 起始帧 结束帧")
            continue
        try:
            label = parts[0]
            s, e = int(parts[1]), int(parts[2])
        except ValueError:
            print("  帧号必须是整数")
            continue
        if e <= s:
            print("  结束帧必须大于起始帧")
            continue
        existing.actions.append(ActionAnnotation(label=label, start_frame=s, end_frame=e))
        print(f"  ✓ 已添加 {label} [{s}, {e})")

    problems = existing.validate()
    if problems:
        print("\n⚠️  自检发现问题:")
        for p in problems:
            print(f"   - {p}")

    videos, _ = load_annotations(out_path) if out_path.exists() else ([], [])
    videos = [v for v in videos if v.video != video]
    videos.append(existing)
    save_annotations(videos, out_path)
    print(f"\n✓ 已保存: {out_path}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """图形标注界面(需要 tkinter)。"""
    try:
        from actionflow.data.annotate_gui import run_gui
    except ImportError as e:
        print(f"无法启动 GUI: {e}")
        print("提示: macOS 用系统 Python 可能有 tkinter,或者用 cli 子命令代替")
        return 1
    return run_gui(args)


def cmd_validate(args: argparse.Namespace) -> int:
    """校验数据集。"""
    from actionflow.data.annotate import validate_dataset

    report = validate_dataset(args.annotations, npz_dir=args.npz_dir)
    return 0 if not report["problems"] else 1


def cmd_export(args: argparse.Namespace) -> int:
    """导出成训练用的 npz 切片。"""
    from actionflow.data.annotate import export_to_npz_dataset, load_annotations

    videos, _ = load_annotations(args.annotations)
    stats = export_to_npz_dataset(
        videos,
        npz_dir=args.npz_dir,
        out_dir=args.out,
        window=args.window,
        stride=args.stride,
    )
    print(f"✓ 导出 {stats['num_windows']} 个窗口到 {args.out}")
    if stats["num_skipped"]:
        print(f"  (跳过 {stats['num_skipped']} 个没有关键点的视频)")
    print("\n  类别分布:")
    total = max(stats["num_windows"], 1)
    for label, c in sorted(stats["label_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {label:16s} {c:5d}  {c/total*100:5.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ActionFlow 数据标注工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    e = sub.add_parser("extract", help="从视频自动抽关键点")
    e.add_argument("--video", required=True)
    e.add_argument("--out", required=True, help="输出 npz")
    e.add_argument("--onnx", default=None, help="姿态模型 ONNX")
    e.add_argument("--threshold", type=float, default=0.3)
    e.set_defaults(func=cmd_extract)

    # gui
    g = sub.add_parser("gui", help="图形标注界面")
    g.add_argument("--video", required=True)
    g.add_argument("--npz", default=None)
    g.add_argument("--out", required=True)
    g.add_argument("--labels", default=None, help="逗号分隔的候选标签")
    g.set_defaults(func=cmd_gui)

    # cli
    c = sub.add_parser("cli", help="命令行标注")
    c.add_argument("--video", required=True)
    c.add_argument("--npz", default=None)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_cli)

    # validate
    v = sub.add_parser("validate", help="校验标注质量")
    v.add_argument("--annotations", required=True, help="标注 JSON 所在目录")
    v.add_argument("--npz-dir", default=None, help="关键点 npz 所在目录")
    v.set_defaults(func=cmd_validate)

    # export
    x = sub.add_parser("export", help="导出训练用 npz 切片")
    x.add_argument("--annotations", required=True)
    x.add_argument("--npz-dir", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--window", type=int, default=32)
    x.add_argument("--stride", type=int, default=16)
    x.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
