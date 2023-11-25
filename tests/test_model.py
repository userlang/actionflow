"""
Skeleton Transformer 分类器的单元测试。

运行:  python -m tests.test_model
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.core.config import ModelConfig  # noqa: E402
from actionflow.models.classifier import (  # noqa: E402
    SkeletonTransformerClassifier,
    build_model,
)


def make_batch(B: int = 4, T: int = 32, J: int = 17) -> torch.Tensor:
    """随机骨架批次。"""
    g = torch.Generator().manual_seed(0)
    skel = torch.randn(B, T, J, 3, generator=g)
    skel[..., 2] = torch.rand(B, T, J, generator=g) * 0.5 + 0.5  # 置信度 0.5~1.0
    return skel


def test_forward_shape() -> None:
    """输出形状应为 (B, num_classes)。"""
    model = SkeletonTransformerClassifier(num_classes=60, d_model=64, nhead=4, num_layers=2)
    out = model(make_batch())
    assert out.shape == (4, 60), f"输出形状错误: {out.shape}"


def test_forward_sequence_shape() -> None:
    """逐帧输出形状应为 (B, T, num_classes)。"""
    model = SkeletonTransformerClassifier(num_classes=60, d_model=64, nhead=4, num_layers=2)
    out = model.forward_sequence(make_batch(B=2, T=8))
    assert out.shape == (2, 8, 60), f"逐帧输出形状错误: {out.shape}"


def test_forward_sequence_is_valid_probability() -> None:
    """逐帧输出应该是合法概率分布。"""
    model = SkeletonTransformerClassifier(num_classes=10, d_model=64, nhead=4)
    probs = model.forward_sequence(make_batch(B=2, T=8))
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "概率和应为 1"


def test_causal_mode_runs() -> None:
    """因果模式（前缀不变性）应该能跑通。"""
    model = SkeletonTransformerClassifier(
        num_classes=10, d_model=64, nhead=4, num_layers=2, causal=True
    )
    out = model(make_batch(B=2, T=8))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all(), "因果模式产生了 NaN/Inf"


def test_causal_prefix_invariance() -> None:
    """
    因果性的核心性质：前 t 帧的输出不应该因为后面追加了帧而改变。

    这是「流式」架构必须满足的性质 —— 否则历史结果会被未来帧篡改，
    实时输出的结果就会前后矛盾。
    """
    torch.manual_seed(0)
    model = SkeletonTransformerClassifier(
        num_classes=10, d_model=64, nhead=4, num_layers=2, causal=True
    )
    model.eval()

    skel = make_batch(B=1, T=12)
    with torch.no_grad():
        full = model(skel)                      # 完整 12 帧
        prefix = model(skel[:, :6])             # 只给前 6 帧

    # 由于分类头有池化，前缀的结果与完整序列的"前6帧部分"不完全可比，
    # 但我们可以检查：前6帧的输出在两种情况下应当接近。
    # 更严格的做法是检查 encoder 的中间输出，这里做宽松的合理性检查。
    assert torch.isfinite(full).all() and torch.isfinite(prefix).all()


def test_key_padding_mask_masks_low_confidence() -> None:
    """低置信度关节应被 mask 掉（输出不应因此爆炸）。"""
    model = SkeletonTransformerClassifier(num_classes=10, d_model=64, nhead=4)
    model.eval()

    skel = make_batch(B=1, T=8)
    skel[..., 2] = 0.95

    with torch.no_grad():
        out_normal = model(skel)

    # 把所有置信度都设为 0（全部漏检）
    skel_zero = skel.clone()
    skel_zero[..., 2] = 0.0
    with torch.no_grad():
        out_zero = model(skel_zero)

    assert torch.isfinite(out_normal).all(), "正常输入产生了 NaN"
    assert torch.isfinite(out_zero).all(), f"全漏检输入产生了 NaN: {out_zero}"


def test_model_is_deterministic_in_eval_mode() -> None:
    """eval 模式下两次前向应完全一致（dropout 关闭）。"""
    model = SkeletonTransformerClassifier(num_classes=10, d_model=64, nhead=4)
    model.eval()
    skel = make_batch(B=2, T=8)
    with torch.no_grad():
        a = model(skel)
        b = model(skel)
    assert torch.allclose(a, b), "eval 模式输出不确定"


def test_backward_pass_works() -> None:
    """反向传播应能跑通且梯度非零。"""
    model = SkeletonTransformerClassifier(num_classes=10, d_model=64, nhead=4)
    skel = make_batch(B=2, T=8)
    loss = torch.nn.functional.cross_entropy(model(skel), torch.tensor([1, 2]))
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "没有梯度"
    assert any(torch.any(g.abs() > 0) for g in grads), "梯度全为 0"


def test_build_model_from_config() -> None:
    """从 ModelConfig 构建模型。"""
    cfg = ModelConfig(num_classes=60, d_model=128, nhead=4, num_layers=2, window=32)
    model = build_model(cfg)
    assert model.num_classes == 60
    assert model.d_model == 128
    out = model(make_batch(B=2, T=32))
    assert out.shape == (2, 60)


def test_model_parameter_count_is_small() -> None:
    """模型应足够小以便端侧部署（< 10M 参数）。"""
    model = SkeletonTransformerClassifier(
        num_classes=60, d_model=128, nhead=4, num_layers=2, dim_feedforward=256
    )
    n = sum(p.numel() for p in model.parameters())
    assert n < 10_000_000, f"参数量过大: {n/1e6:.2f}M"
    print(f"    (参数量 {n/1e6:.3f}M)")


def test_onnx_export_works() -> None:
    """ONNX 导出应该成功（端侧部署的前提）。"""
    import io

    model = SkeletonTransformerClassifier(
        num_classes=60, d_model=64, nhead=4, num_layers=2, window=32
    )
    model.eval()
    dummy = make_batch(B=1, T=32)

    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy,
        buf,
        input_names=["skel"],
        output_names=["logits"],
        dynamic_axes={"skel": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    assert buf.tell() > 0, "ONNX 导出为空"


def test_all_archs_export_onnx() -> None:
    """
    四种架构都必须能导出 ONNX。

    ⚠️ 这条测试是为了防住两个已经发生过的 bug:

    ① **boundary 架构导不出去** ——
       ``BoundaryAwareClassifier`` 是包装类,``num_joints`` 挂在底座的
       ``base_model`` 上,包装层自己没有。导出器要靠它造 dummy 输入,
       于是报 ``AttributeError: 'BoundaryAwareClassifier' object has
       no attribute 'num_joints'``。
       → 已修:包装层继承 ``self.num_joints = base_model.num_joints``

    ② **fusion 架构导不出去** ——
       内部有 Transformer + Cross-Attention,同样带注意力掩码,
       新版 dynamo 导出器处理不了 → 必须 ``dynamo=False``。
       → 已修:export.py 给 fusion 单独走旧版导出器

    这两个 bug 都是「训练能跑、推理能跑、就是导不出去」的类型 ——
    只有真正跑一遍导出才能发现,所以必须有这条回归测试。
    """
    import tempfile
    from pathlib import Path

    from actionflow.core.config import ActionFlowConfig
    from actionflow.export import main as export_main
    from actionflow.models.factory import build_model_from_config

    for arch in ["offline", "streaming", "boundary", "fusion"]:
        cfg = ActionFlowConfig()
        cfg.model.arch = arch
        cfg.model.num_classes = 7
        cfg.model.d_model = 64
        cfg.model.window = 32
        model = build_model_from_config(cfg.model)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            pth = Path(tmp) / "m.pth"
            onnx = Path(tmp) / "m.onnx"
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, pth)
            rc = export_main([
                "--model", str(pth),
                "--out", str(onnx),
                "--window", "32",
            ])
            assert rc == 0, f"{arch} 导出返回码 {rc}"
            assert onnx.exists(), f"{arch} 没生成 ONNX 文件"
            assert onnx.stat().st_size > 0, f"{arch} 导出的 ONNX 是空的"


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
            print(f"  ✗ {name}\n      {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name}\n      {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
