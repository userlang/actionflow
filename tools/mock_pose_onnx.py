"""
生成一个「假的」RTMPose ONNX 模型 —— 仅用于在没有真实权重的环境下
验证整条流水线（预处理 → 推理 → SimCC 解码 → 后处理）能跑通。

这个模型不做真正的姿态估计，它根据图像里有没有人形图案返回随机的
SimCC 分布。**仅供测试，不可用于实际识别。**

用法:
    python tools/mock_pose_onnx.py --out weights/mock-pose.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_mock_onnx(out_path: str | Path, num_joints: int = 17, size: int = 192) -> None:
    """构造一个输出随机 SimCC 分布的 ONNX 模型。"""
    import torch
    import torch.nn as nn

    class MockSimCC(nn.Module):
        """
        模拟 RTMPose 的 SimCC 双头输出。

        为了让输出像真实模型一样是**尖峰分布**（而不是均匀分布），
        这里在每个关节的 1D 分布上叠加一个高斯峰。峰值高度随机，
        模拟不同关键点的不同置信度。
        """

        def __init__(self, num_joints: int, size: int) -> None:
            super().__init__()
            self.num_joints = num_joints
            self.size = size
            # 每个关节的峰值位置和强度（固定，随机初始化）
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("peak_x", torch.rand(num_joints, generator=g) * size)
            self.register_buffer("peak_y", torch.rand(num_joints, generator=g) * size)
            # 峰值强度与锐度：模拟真实 RTMPose 的 SimCC 输出。
            # 真实模型的分布很尖（softmax 后峰值 0.3~0.9），
            # sigma 取 size/32 左右、峰高 30 可复现这个量级。
            self.register_buffer("peak_h", torch.rand(num_joints, generator=g) * 8.0 + 26.0)
            sigma = size / 32.0
            self.register_buffer("sigma", torch.tensor(sigma))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (B, 3, H, W) —— 这里只用 batch 维，内容不参与计算
            B = x.shape[0]
            device = x.device

            arange = torch.arange(self.size, device=device, dtype=torch.float32)

            # x 方向高斯峰: (J, W)
            gx = torch.exp(-((arange.view(1, -1) - self.peak_x.view(-1, 1)) ** 2) / (2 * self.sigma**2))
            gx = gx * self.peak_h.view(-1, 1)

            # y 方向高斯峰: (J, H)
            gy = torch.exp(-((arange.view(1, -1) - self.peak_y.view(-1, 1)) ** 2) / (2 * self.sigma**2))
            gy = gy * self.peak_h.view(-1, 1)

            logits_x = gx.unsqueeze(0).expand(B, -1, -1)
            logits_y = gy.unsqueeze(0).expand(B, -1, -1)

            return logits_x, logits_y

    model = MockSimCC(num_joints, size)
    model.eval()

    dummy = torch.randn(1, 3, size, size)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["simcc_x", "simcc_y"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch"},
            "simcc_x": {0: "batch"},
            "simcc_y": {0: "batch"},
        },
    )
    print(f"✓ 已生成 mock ONNX: {out_path}")
    print("  ⚠️  这是一个用于测试的假模型，不能做真正的姿态估计")


def main() -> int:
    p = argparse.ArgumentParser(description="生成 mock RTMPose ONNX（仅测试用）")
    p.add_argument("--out", default="weights/mock-pose.onnx")
    args = p.parse_args()
    build_mock_onnx(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
