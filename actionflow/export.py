"""
ONNX 导出 —— 端侧部署的前提。

用法:
    python -m actionflow.export --model weights/actionflow_skel.pth --out weights/af.onnx

导出后可用 onnxruntime 直接推理，不需要 PyTorch。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 ActionFlow 分类器为 ONNX")
    p.add_argument("--model", required=True, help="训练好的 .pth 权重")
    p.add_argument("--out", default="weights/actionflow_skel.onnx", help="输出 ONNX 路径")
    p.add_argument("--window", type=int, default=32, help="窗口长度（动态轴，导出时用的示例值）")
    # opset 18+ 是 torch 2.15 / onnxscript 新导出器的原生版本；
    # 设成 17 会触发降级转换器并打出大量无关警告。
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--dynamic-time", action="store_true", default=True, help="时间维是否动态")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    from actionflow.core.config import ActionFlowConfig
    from actionflow.models.factory import build_model_from_config

    args = parse_args(argv)

    ckpt = torch.load(args.model, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    # 优先用权重里保存的配置
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ActionFlowConfig.from_dict(ckpt["config"])
    else:
        cfg = ActionFlowConfig()

    model = build_model_from_config(cfg.model)
    model.load_state_dict(state)
    model.eval()

    num_joints = cfg.model.num_joints
    dummy = torch.randn(1, args.window, num_joints, 3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 流式模型要用专用的导出路径 ──────────────────────────────
    # 原因见 actionflow/models/streaming.py 的 export_streaming_onnx:
    # 新版 dynamo 导出器处理不了注意力掩码,必须用旧版导出器(dynamo=False),
    # 而且窗口长度必须静态。
    arch = getattr(cfg.model, "arch", "streaming")
    if arch in ("streaming", "boundary"):
        from actionflow.models.streaming import export_streaming_onnx

        path = export_streaming_onnx(model, str(out_path), window=args.window)
        print(f"✓ 导出成功: {path}")
        print(f"  架构: {arch}（专用导出器,窗口长度固定为 {args.window}）")
        _report_size(model, path)
        _verify_onnx(path, num_joints, args.window)
        return 0

    dynamic_axes = {"skel": {0: "batch", 1: "time"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["skel"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    _report_size(model, out_path)
    _verify_onnx(out_path, num_joints, args.window)
    return 0


def _report_size(model, path) -> None:
    """打印参数量和文件大小。"""
    size_mb = Path(path).stat().st_size / 1024 / 1024
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 导出成功: {path}")
    print(f"  参数量: {n_params/1e6:.3f}M")
    print(f"  文件大小: {size_mb:.2f} MB")


def _verify_onnx(path, num_joints: int, window: int) -> None:
    """
    验证导出的模型真能跑。

    这一步不能省 —— 导出"成功"不等于"结果正确"。
    实测中出现过导出过程不报错、但 onnxruntime 加载后输出全错的情况。
    """
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        test = np.random.randn(1, window, num_joints, 3).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: test})
        print(f"  ONNX 推理验证通过, 输出 shape={out[0].shape}")
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] ONNX 验证失败: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
