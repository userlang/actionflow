"""
因果流式模型的单元测试。

运行:  python tests/test_streaming.py

这些测试验证的是流式架构最核心的两条性质:
  1. 因果掩码的**语义正确**（帧内可见、帧间不可见）
  2. **前缀不变性** —— 这是流式架构的立身之本
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.models.streaming import (  # noqa: E402
    CausalStreamingClassifier,
    build_causal_frame_mask,
    build_streaming_model,
)


def make_batch(B: int = 2, T: int = 16, J: int = 17) -> torch.Tensor:
    """随机骨架批次(置信度 0.5~1.0,确保不漏检)。"""
    g = torch.Generator().manual_seed(0)
    skel = torch.randn(B, T, J, 3, generator=g)
    skel[..., 2] = torch.rand(B, T, J, generator=g) * 0.5 + 0.5
    return skel


# ===========================================================================
# 掩码语义
# ===========================================================================
def test_causal_mask_shape() -> None:
    """掩码形状应为 (T*J, T*J)。"""
    m = build_causal_frame_mask(num_frames=4, num_joints=17)
    assert m.shape == (68, 68), f"掩码形状错误: {m.shape}"
    assert m.dtype == torch.bool, "掩码必须是 bool 类型"


def test_causal_mask_allows_same_frame() -> None:
    """
    同一帧内的关节必须互相可见。

    这是最容易搞错的地方 —— 右手腕要看右手肘才能判断抬手,
    如果被掩码挡住,模型就永远学不会"抬手"。
    """
    J, T = 17, 4
    m = build_causal_frame_mask(T, J)
    for t in range(T):
        for j in range(J):
            for j2 in range(J):
                i = t * J + j
                k = t * J + j2
                assert not bool(m[i, k]), f"同帧 token {i}→{k} 被错误屏蔽"


def test_causal_mask_blocks_future_frames() -> None:
    """不能看到未来帧的 token。"""
    J, T = 17, 4
    m = build_causal_frame_mask(T, J)
    for t in range(T):
        for t2 in range(t + 1, T):
            i = t * J
            k = t2 * J
            assert bool(m[i, k]), f"未来帧 {t2} 竟被 token {i} 看到"


def test_causal_mask_allows_past_frames() -> None:
    """必须能看到过去帧的 token(不然就退化成只看当前帧)。"""
    J, T = 17, 4
    m = build_causal_frame_mask(T, J)
    for t in range(T):
        for t2 in range(t):
            i = t * J
            k = t2 * J
            assert not bool(m[i, k]), f"过去帧 {t2} 被 token {i} 屏蔽了"


def test_causal_mask_is_lower_triangular_by_frame() -> None:
    """
    整体形状:每一行里,"未来帧"的部分必须是 True(禁止),其余是 False。

    这是把上面三条规则合起来看的整体断言。
    """
    J, T = 17, 5
    m = build_causal_frame_mask(T, J)
    for i in range(T * J):
        ti = i // J
        for j in range(T * J):
            tj = j // J
            expected = tj > ti
            assert bool(m[i, j]) == expected, f"mask[{i},{j}] 应为 {expected}"


def test_causal_mask_independent_of_length() -> None:
    """
    掩码的前缀不受总长度影响 —— 这是滑窗推理安全的前提。

    如果前 8 帧的掩码和 16 帧版本的前 8 帧部分不一样,
    那么窗口滑动时历史结果就会变,流式就不可信了。
    """
    J = 17
    m8 = build_causal_frame_mask(8, J)
    m16 = build_causal_frame_mask(16, J)
    assert torch.equal(m8, m16[: 8 * J, : 8 * J]), "掩码前缀随长度变化了"


# ===========================================================================
# 前缀不变性 —— 流式架构的立身之本
# ===========================================================================
def test_prefix_invariance_exact() -> None:
    """
    【核心测试】只用前 t 帧算出的结果,必须和用更多帧算出的前 t 帧结果**完全一致**。

    为什么这条如此重要:
    实时系统里,第 10 帧的结论一旦输出就无法撤回。
    如果后面的帧能改变历史结论,那系统输出就会自相矛盾。
    """
    torch.manual_seed(0)
    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4, num_layers=2)
    model.eval()

    skel = make_batch(B=1, T=16)
    with torch.no_grad():
        full = model(skel)                    # (1, 16, 10)
        prefix = model(skel[:, :8])           # (1, 8, 10)

    diff = (full[:, :8] - prefix).abs().max().item()
    assert diff < 1e-5, f"前缀不变性被破坏,最大差异 {diff:.3e}"


def test_prefix_invariance_all_prefixes() -> None:
    """逐个前缀都检查一遍,确保没有"某个特定长度"出问题。"""
    torch.manual_seed(1)
    model = CausalStreamingClassifier(num_classes=8, d_model=64, nhead=4, num_layers=2)
    model.eval()

    skel = make_batch(B=1, T=12)
    with torch.no_grad():
        full = model(skel)
        for t in range(1, 12):
            prefix = model(skel[:, :t])
            diff = (full[:, :t] - prefix).abs().max().item()
            assert diff < 1e-5, f"长度 {t} 处前缀不变性破坏,差异 {diff:.3e}"


def test_prefix_invariance_with_dropout_off() -> None:
    """eval 模式下必须确定性 —— 否则「同一个输入两次结果不同」就没法验收。"""
    torch.manual_seed(2)
    model = CausalStreamingClassifier(num_classes=8, d_model=64, nhead=4, num_layers=2)
    model.eval()

    skel = make_batch(B=1, T=10)
    with torch.no_grad():
        a = model(skel)
        b = model(skel)
    assert torch.allclose(a, b, atol=1e-6), "eval 模式输出不确定"


# ===========================================================================
# 输出形状与数值
# ===========================================================================
def test_streaming_output_shape() -> None:
    """输出形状应为 (B, T, num_classes)。"""
    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    out = model(make_batch(B=2, T=8))
    assert out.shape == (2, 8, 10), f"输出形状错误: {out.shape}"


def test_streaming_probs_are_valid_distribution() -> None:
    """每帧的概率应该加起来等于 1。"""
    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    probs = model.forward_probs(make_batch(B=2, T=8))
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "概率和不为 1"


def test_streaming_is_finite() -> None:
    """不能出现 NaN / Inf。"""
    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    out = model(make_batch(B=2, T=8))
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"


def test_streaming_handles_all_masked_frame() -> None:
    """
    某一帧所有关节都漏检时,不能崩。

    实际场景:人转身后侧身,姿态模型可能一个点都检不出来。
    """
    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    skel = make_batch(B=1, T=8)
    skel[0, 3, :, 2] = 0.0     # 第 3 帧全部关节置信度置 0
    out = model(skel)
    assert torch.isfinite(out).all(), "全漏检帧导致 NaN"


def test_streaming_backward_works() -> None:
    """反向传播要能跑通,否则没法训练。"""
    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    out = model(make_batch(B=2, T=8))
    out.mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "没有任何参数拿到梯度"


def test_build_streaming_model_from_config() -> None:
    """从 ModelConfig 构建。"""
    from actionflow.core.config import ModelConfig

    model = build_streaming_model(ModelConfig(num_classes=12, d_model=64, nhead=4))
    out = model(make_batch(B=1, T=6))
    assert out.shape == (1, 6, 12)


def test_streaming_param_count_is_small() -> None:
    """端侧部署要求参数量小。"""
    model = CausalStreamingClassifier(num_classes=60, d_model=128, nhead=4, num_layers=2)
    n = sum(p.numel() for p in model.parameters())
    assert n < 10_000_000, f"参数量过大: {n/1e6:.2f}M"
    print(f"    (参数量 {n/1e6:.3f}M)")


def test_streaming_onnx_export() -> None:
    """
    ONNX 导出要能过 —— 端侧部署的前提。

    注意:流式模型必须用 `export_streaming_onnx`(dynamo=False),
    因为新版导出器处理不了注意力掩码。详见该函数的说明。
    """
    import tempfile
    from pathlib import Path

    from actionflow.models.streaming import export_streaming_onnx

    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4, num_layers=2, window=8)
    with tempfile.TemporaryDirectory() as tmp:
        path = export_streaming_onnx(model, str(Path(tmp) / "m.onnx"), window=8)
        assert Path(path).stat().st_size > 0, "ONNX 导出为空"


def test_streaming_onnx_output_matches_pytorch() -> None:
    """
    导出的 ONNX 与 PyTorch 的输出必须一致。

    这是导出流程最有价值的检查 —— 导出了但结果对不上,
    等于导出了一个"看起来能跑但结果全错"的模型。
    """
    import tempfile
    from pathlib import Path

    import numpy as np

    from actionflow.models.streaming import export_streaming_onnx

    try:
        import onnxruntime as ort
    except ImportError:
        print("    (跳过: 未安装 onnxruntime)")
        return

    torch.manual_seed(0)
    model = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4, num_layers=2, window=8)
    model.eval()
    skel = make_batch(B=1, T=8)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_streaming_onnx(model, str(Path(tmp) / "m.onnx"), window=8)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"skel": skel.numpy().astype(np.float32)})[0]

    with torch.no_grad():
        torch_out = model(skel).numpy()

    diff = np.abs(onnx_out - torch_out).max()
    assert diff < 1e-4, f"ONNX 与 PyTorch 输出不一致,最大差异 {diff:.3e}"


def test_streaming_matches_naive_prefix_reference() -> None:
    """
    流式模型 vs 朴素前缀实现 —— 两者的**第 t 帧**结果应当非常接近。

    ⚠️ 注意:这里比较的是**同一个模型**的两种调用方式,
       但流式模型和普通模型的**参数结构不同**(逐帧池化 vs 全局池化),
       所以不能直接比较两个不同模型的输出。

       真正的正确性验证是前缀不变性(见上面的测试)。
       本测试只验证流式模型内部的「帧 t 的输出确实只依赖前 t 帧」这一性质,
       通过与"只喂前 t 帧"的调用方式对比。
    """
    torch.manual_seed(3)
    model = CausalStreamingClassifier(num_classes=8, d_model=64, nhead=4, num_layers=2)
    model.eval()

    skel = make_batch(B=1, T=10)
    with torch.no_grad():
        full = model(skel)
        # 第 5 帧单独看(长度 5)
        upto5 = model(skel[:, :5])
    diff = (full[:, 4] - upto5[:, 4]).abs().max().item()
    assert diff < 1e-5, f"第5帧依赖了未来信息,差异 {diff:.3e}"


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
