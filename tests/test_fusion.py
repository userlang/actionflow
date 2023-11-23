"""
双路融合模型的单元测试。

运行:  python tests/test_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.models.fusion import (  # noqa: E402
    CrossAttentionFusion,
    FusionClassifier,
    SimpleVideoBackbone,
    VideoBackbone,
)
from actionflow.models.streaming import CausalStreamingClassifier  # noqa: E402


def make_skel(B: int = 2, T: int = 8, J: int = 17) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    skel = torch.randn(B, T, J, 3, generator=g)
    skel[..., 2] = torch.rand(B, T, J, generator=g) * 0.5 + 0.5
    return skel


def make_frames(B: int = 2, T: int = 8, H: int = 32, W: int = 32) -> torch.Tensor:
    return torch.randn(B, T, 3, H, W)


# ===========================================================================
# 视频 backbone
# ===========================================================================
def test_video_backbone_is_abstract() -> None:
    """抽象基类不能直接实例化使用。"""
    backbone = VideoBackbone()
    try:
        backbone(torch.randn(1, 4, 3, 32, 32))
        raise AssertionError("抽象方法应该抛 NotImplementedError")
    except NotImplementedError:
        pass


def test_simple_video_backbone_shape() -> None:
    """输出应为 (B, T, out_dim)。"""
    backbone = SimpleVideoBackbone(out_dim=64)
    out = backbone(make_frames(B=2, T=4))
    assert out.shape == (2, 4, 64), f"输出形状错误: {out.shape}"


def test_simple_video_backbone_is_differentiable() -> None:
    """梯度要能回传到输入。"""
    backbone = SimpleVideoBackbone(out_dim=32)
    frames = torch.randn(1, 3, 3, 32, 32, requires_grad=True)
    out = backbone(frames)
    out.sum().backward()
    assert frames.grad is not None and frames.grad.abs().sum() > 0, "梯度没有回传"


def test_simple_video_backbone_detects_motion() -> None:
    """
    3D 卷积应该能区分"静态"和"运动"。

    这是视频路存在的意义 —— 骨架路做不到的事。
    两段视频:同样的 3 帧,但一张是静止的,一张是移动的,
    特征不应该完全一样。
    """
    torch.manual_seed(0)
    backbone = SimpleVideoBackbone(out_dim=32)
    backbone.eval()

    static = torch.ones(1, 4, 3, 32, 32)
    moving = torch.stack([torch.full((3, 32, 32), float(i)) for i in range(4)]).unsqueeze(0)

    with torch.no_grad():
        a = backbone(static)
        b = backbone(moving)

    diff = (a - b).abs().max().item()
    assert diff > 1e-4, "视频 backbone 分不清静态和运动(3D 卷积没起作用)"


# ===========================================================================
# 交叉注意力
# ===========================================================================
def test_cross_attention_shape() -> None:
    """输出形状应为 (B, T, d_model)。"""
    fusion = CrossAttentionFusion(d_skel=64, d_video=32, d_model=64, nhead=4)
    s = torch.randn(2, 8, 64)
    v = torch.randn(2, 8, 32)
    fs, fv = fusion(s, v)
    assert fs.shape == (2, 8, 64), f"骨架融合形状错误: {fs.shape}"
    assert fv.shape == (2, 8, 64), f"视频融合形状错误: {fv.shape}"


def test_cross_attention_is_causal() -> None:
    """
    交叉注意力本身必须是因果的。

    ⚠️ 这是个很容易漏的坑。
       整个融合模型有两处需要因果掩码:
           ① CrossAttentionFusion 内部的交叉注意力
           ② post_fusion_encoder 这一层 TransformerEncoder
       只加 ② 的话,第 3 帧照样会去看第 8 帧的视频特征 ——
       前缀不变性直接崩掉(实测差异 2.2e-05)。

    这个测试单独盯住 ①,不依赖 ②。
    """
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(d_skel=64, d_video=64, d_model=64, nhead=4)
    fusion.eval()

    s = torch.randn(1, 8, 64)
    v = torch.randn(1, 8, 64)

    with torch.no_grad():
        full, _ = fusion(s, v, causal=True)
        pre, _ = fusion(s[:, :4], v[:, :4], causal=True)

    diff = (full[:, :4] - pre).abs().max().item()
    assert diff < 1e-5, f"交叉注意力破坏了因果性,差异 {diff:.3e}"


def test_cross_attention_bidirectional_when_not_causal() -> None:
    """
    causal=False 时应该恢复成双向(离线推理用)。

    顺便验证这个开关真的起作用 —— 不是摆设。
    """
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(d_skel=64, d_video=64, d_model=64, nhead=4)
    fusion.eval()

    s = torch.randn(1, 8, 64)
    v = torch.randn(1, 8, 64)

    with torch.no_grad():
        full_c, _ = fusion(s, v, causal=True)
        pre_c, _ = fusion(s[:, :4], v[:, :4], causal=True)
        full_b, _ = fusion(s, v, causal=False)
        pre_b, _ = fusion(s[:, :4], v[:, :4], causal=False)

    diff_c = (full_c[:, :4] - pre_c).abs().max().item()
    diff_b = (full_b[:, :4] - pre_b).abs().max().item()

    assert diff_c < 1e-5, f"causal=True 时应保持因果,差异 {diff_c:.3e}"
    assert diff_b > 1e-4, f"causal=False 时应允许看未来,差异 {diff_b:.3e}"


def test_cross_attention_uses_residual() -> None:
    """
    残差连接:融合结果应该"接近"原始骨架特征,而不是被完全替换。

    这是"锦上添花"而不是"喧宾夺主"的关键。

    ⚠️ 注意不能直接比 cosine similarity ——
       因为融合后还套了一层 LayerNorm,量纲被归一化了,
       和原始特征不在同一个尺度上。

    正确的测法:**把注意力分支"关掉"**(权重置零),
    看融合结果是否等于原始骨架特征。
    如果关掉后结果变了,说明残差路径确实在起作用。
    """
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(d_skel=64, d_video=64, d_model=64, nhead=4)
    fusion.eval()

    s = torch.randn(1, 8, 64)
    v = torch.randn(1, 8, 64)

    with torch.no_grad():
        fs_on, _ = fusion(s, v)

        # 把「注意力分支」的输出投影清零 → 等价于只剩残差路径
        for p in (fusion.out_skel.weight, fusion.out_skel.bias):
            p.zero_()

        fs_off, _ = fusion(s, v)

    # 关掉注意力分支后,融合结果 = LayerNorm(投影后的骨架)
    # 注意是"投影后"——融合模块内部先把 d_skel 维投到 d_model 维,
    # 残差加的是投影后的结果,不是原始输入。
    expected = fusion.norm_skel(fusion.skel_proj(s))
    diff_off = (fs_off - expected).abs().max().item()
    assert diff_off < 1e-5, f"残差路径不对: 差异 {diff_off:.3e}"

    # 开着注意力分支时,结果应该和只有残差时不同(说明确实加进去了东西)
    diff_on = (fs_on - fs_off).abs().max().item()
    assert diff_on > 1e-4, "注意力分支没有起作用(融合结果和纯残差一样)"


def test_cross_attention_is_differentiable() -> None:
    """梯度要能回传到两条路。"""
    fusion = CrossAttentionFusion(d_skel=32, d_video=48, d_model=32, nhead=4)
    s = torch.randn(1, 6, 32, requires_grad=True)
    v = torch.randn(1, 6, 48, requires_grad=True)
    fs, fv = fusion(s, v)
    (fs.sum() + fv.sum()).backward()
    assert s.grad is not None and s.grad.abs().sum() > 0, "骨架路没拿到梯度"
    assert v.grad is not None and v.grad.abs().sum() > 0, "视频路没拿到梯度"


def test_cross_attention_head_dim_divides() -> None:
    """d_model 必须能被 nhead 整除。"""
    fusion = CrossAttentionFusion(d_skel=64, d_video=64, d_model=64, nhead=8)
    s = torch.randn(1, 4, 64)
    v = torch.randn(1, 4, 64)
    fs, _ = fusion(s, v)
    assert fs.shape == (1, 4, 64)


# ===========================================================================
# 融合分类器
# ===========================================================================
def test_fusion_classifier_output_shape() -> None:
    """输出 (B, T, C)。"""
    base = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4, num_layers=2)
    backbone = SimpleVideoBackbone(out_dim=64)
    model = FusionClassifier(base, backbone, d_video=64, nhead=4)
    out = model(make_skel(B=2, T=6), make_frames(B=2, T=6))
    assert out.shape == (2, 6, 10), f"输出形状错误: {out.shape}"


def test_fusion_classifier_without_video() -> None:
    """不给视频帧时,只走骨架路(视频路是可选的)。"""
    base = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    backbone = SimpleVideoBackbone(out_dim=64)
    model = FusionClassifier(base, backbone, d_video=64, nhead=4, use_video=True)
    out = model(make_skel(B=1, T=6), None)
    assert out.shape == (1, 6, 10)
    assert torch.isfinite(out).all()


def test_fusion_classifier_backward() -> None:
    """两条路都要能拿到梯度。"""
    base = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    backbone = SimpleVideoBackbone(out_dim=64)
    model = FusionClassifier(base, backbone, d_video=64, nhead=4)

    skel = make_skel(B=1, T=4)
    frames = make_frames(B=1, T=4)
    out = model(skel, frames)
    out.mean().backward()

    assert backbone.net[0].weight.grad is not None, "视频 backbone 没拿到梯度"
    assert model.base.classifier_head[1].weight.grad is not None, "分类头没拿到梯度"


def test_fusion_classifier_keeps_prefix_invariance() -> None:
    """
    融合之后前缀不变性**不能被破坏**。

    视频路如果破坏了因果性,流式就废了。

    ⚠️ 这里有**两处**需要因果掩码,少一处就崩:
        ① CrossAttentionFusion 内部的交叉注意力
        ② post_fusion_encoder 这一层 TransformerEncoder
    只加 ② 的话,交叉注意力还是双向的,
    第 3 帧照样会去看第 8 帧的视频特征。
    """
    torch.manual_seed(0)
    base = CausalStreamingClassifier(num_classes=8, d_model=64, nhead=4, num_layers=2)
    backbone = SimpleVideoBackbone(out_dim=64)
    model = FusionClassifier(base, backbone, d_video=64, nhead=4)
    model.eval()

    skel = make_skel(B=1, T=12)
    frames = make_frames(B=1, T=12)
    with torch.no_grad():
        full = model(skel, frames)
        prefix = model(skel[:, :6], frames[:, :6])

    diff = (full[:, :6] - prefix).abs().max().item()
    assert diff < 1e-5, f"融合破坏了前缀不变性,差异 {diff:.3e}"


def test_fusion_classifier_param_count() -> None:
    """双路模型参数量应该仍然可控(视频 backbone 是轻量的)。"""
    base = CausalStreamingClassifier(num_classes=60, d_model=128, nhead=4, num_layers=2)
    backbone = SimpleVideoBackbone(out_dim=128)
    model = FusionClassifier(base, backbone, d_video=128, nhead=4)
    n = sum(p.numel() for p in model.parameters())
    print(f"    (参数量 {n/1e6:.3f}M)")
    assert n < 30_000_000, f"参数量过大: {n/1e6:.2f}M"


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
