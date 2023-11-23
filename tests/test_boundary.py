"""
边界回归头的单元测试。

运行:  python tests/test_boundary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.models.boundary import (  # noqa: E402
    BoundaryAwareClassifier,
    BoundaryRegressionHead,
    boundary_regression_loss,
    build_boundary_aware_model,
    compute_boundary_targets,
)
from actionflow.models.streaming import CausalStreamingClassifier  # noqa: E402
from actionflow.postprocess.decode import (  # noqa: E402
    ActionSegment,
    decode_segments,
    refine_boundaries_with_regression,
)


def make_batch(B: int = 2, T: int = 16, J: int = 17) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    skel = torch.randn(B, T, J, 3, generator=g)
    skel[..., 2] = torch.rand(B, T, J, generator=g) * 0.5 + 0.5
    return skel


# ===========================================================================
# 标签构造
# ===========================================================================
def test_boundary_targets_simple() -> None:
    """
    最简单的情形:A 段 5 帧,B 段 7 帧。

    预期:
        距起点: 0 1 2 3 4 | 0 1 2 3 4 5 6
        距终点: 4 3 2 1 0 | 6 5 4 3 2 1 0
    """
    labels = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
    d_start, d_end = compute_boundary_targets(labels, num_classes=2)

    assert d_start.shape == (1, 12) and d_end.shape == (1, 12)
    exp_start = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]
    exp_end = [4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0]
    assert d_start[0].tolist() == exp_start, f"距起点错误: {d_start[0].tolist()}"
    assert d_end[0].tolist() == exp_end, f"距终点错误: {d_end[0].tolist()}"


def test_boundary_targets_single_segment() -> None:
    """只有一段时,距终点应该一路递减到 0。"""
    labels = torch.tensor([[2, 2, 2, 2]])
    d_start, d_end = compute_boundary_targets(labels, num_classes=3)
    assert d_start[0].tolist() == [0, 1, 2, 3]
    assert d_end[0].tolist() == [3, 2, 1, 0]


def test_boundary_targets_all_different() -> None:
    """每帧类别都不同 → 所有距离都是 0(每帧自己就是边界)。"""
    labels = torch.tensor([[0, 1, 2, 3]])
    d_start, d_end = compute_boundary_targets(labels, num_classes=4)
    assert d_start[0].tolist() == [0, 0, 0, 0]
    assert d_end[0].tolist() == [0, 0, 0, 0]


def test_boundary_targets_batch() -> None:
    """批量输入,每个样本独立计算。"""
    labels = torch.tensor([
        [0, 0, 0, 1, 1],
        [5, 5, 5, 5, 5],
    ])
    d_start, d_end = compute_boundary_targets(labels, num_classes=6)
    assert d_start.shape == (2, 5)
    assert d_start[1].tolist() == [0, 1, 2, 3, 4], "第二个样本应独立计算"


# ===========================================================================
# 回归头
# ===========================================================================
def test_boundary_head_output_shape() -> None:
    """输出形状应为 (B, T, 2)。"""
    head = BoundaryRegressionHead(d_model=64)
    out = head(torch.randn(2, 10, 64))
    assert out.shape == (2, 10, 2), f"输出形状错误: {out.shape}"


def test_boundary_head_is_non_negative() -> None:
    """距离不可能是负数 —— softplus 保证这一点。"""
    head = BoundaryRegressionHead(d_model=64)
    out = head(torch.randn(4, 20, 64) * 10)   # 故意喂极端值
    assert (out >= 0).all(), "softplus 之后仍出现负数"


def test_boundary_head_is_differentiable_everywhere() -> None:
    """
    softplus 相比 ReLU 的关键优势:处处可导。

    ReLU 在输入为负时梯度直接是 0,模型就"卡住"学不动了。
    softplus 的梯度永远大于 0。
    """
    head = BoundaryRegressionHead(d_model=8)
    x = torch.randn(1, 5, 8, requires_grad=True)
    out = head(x)
    out.sum().backward()
    assert (x.grad.abs() > 0).all(), "存在梯度为 0 的输入点(softplus 不该这样)"


# ===========================================================================
# 损失函数
# ===========================================================================
def test_boundary_loss_zero_when_perfect() -> None:
    """预测完全正确时损失应为 0。"""
    target = torch.tensor([[[1.0, 2.0], [0.0, 1.0]]])
    loss = boundary_regression_loss(target.clone(), target)
    assert loss.item() < 1e-6, f"完美预测时损失应为 0,实际 {loss.item()}"


def test_boundary_loss_is_l1() -> None:
    """确认是 L1(绝对值)而不是 L2(平方)。"""
    pred = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[2.0, 2.0]]])
    loss = boundary_regression_loss(pred, target)
    # L1: 平均(|0-2|) = 2.0;L2: 平均((0-2)^2) = 4.0
    assert abs(loss.item() - 2.0) < 1e-5, f"L1 损失应为 2.0,实际 {loss.item()}"


def test_boundary_loss_respects_mask() -> None:
    """被 mask 掉的帧不应参与损失。"""
    pred = torch.tensor([[[0.0, 0.0], [100.0, 100.0]]])
    target = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    masked = boundary_regression_loss(pred, target, mask)
    unmasked = boundary_regression_loss(pred, target)
    assert masked.item() < unmasked.item(), "mask 没有起到屏蔽作用"
    assert masked.item() < 1e-6, "被屏蔽帧不应产生损失"


def test_boundary_loss_mask_all_invalid() -> None:
    """全 mask 时不能除零崩掉。"""
    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    loss = boundary_regression_loss(pred, target, mask)
    assert torch.isfinite(loss), "全 mask 时损失应为有限值"


# ===========================================================================
# 组合模型
# ===========================================================================
def test_boundary_aware_model_shapes() -> None:
    """应同时输出 (B,T,C) 的 logits 和 (B,T,2) 的距离。"""
    base = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4, num_layers=2)
    model = BoundaryAwareClassifier(base, use_boundary=True)
    logits, dist = model(make_batch(B=2, T=8))
    assert logits.shape == (2, 8, 10), f"logits 形状错误: {logits.shape}"
    assert dist.shape == (2, 8, 2), f"距离形状错误: {dist.shape}"


def test_boundary_aware_model_without_boundary_head() -> None:
    """关闭边界头时,距离应为 None。"""
    base = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    model = BoundaryAwareClassifier(base, use_boundary=False)
    logits, dist = model(make_batch(B=1, T=6))
    assert dist is None
    assert logits.shape == (1, 6, 10)


def test_boundary_aware_model_backward() -> None:
    """两个头的梯度都要能回传。"""
    base = CausalStreamingClassifier(num_classes=10, d_model=64, nhead=4)
    model = BoundaryAwareClassifier(base, use_boundary=True)
    logits, dist = model(make_batch(B=2, T=8))
    (logits.mean() + dist.mean()).backward()

    assert model.boundary_head.mlp[0].weight.grad is not None, "边界头没拿到梯度"
    assert model.base.classifier_head[1].weight.grad is not None, "分类头没拿到梯度"


def test_build_boundary_aware_model_from_config() -> None:
    """从 ModelConfig 构建。"""
    from actionflow.core.config import ModelConfig

    model = build_boundary_aware_model(ModelConfig(num_classes=12, d_model=64, nhead=4))
    logits, dist = model(make_batch(B=1, T=6))
    assert logits.shape == (1, 6, 12)
    assert dist.shape == (1, 6, 2)


def test_boundary_model_keeps_prefix_invariance() -> None:
    """
    加了边界头之后,前缀不变性**不能被破坏**。

    这是个容易被忽略的坑:如果边界头引入了全局池化,
    就会把未来的信息泄漏到历史帧里,流式就废了。
    """
    torch.manual_seed(0)
    base = CausalStreamingClassifier(num_classes=8, d_model=64, nhead=4, num_layers=2)
    model = BoundaryAwareClassifier(base, use_boundary=True)
    model.eval()

    skel = make_batch(B=1, T=16)
    with torch.no_grad():
        full_logits, full_dist = model(skel)
        pre_logits, pre_dist = model(skel[:, :8])

    d1 = (full_logits[:, :8] - pre_logits).abs().max().item()
    d2 = (full_dist[:, :8] - pre_dist).abs().max().item()
    assert d1 < 1e-5, f"分类头前缀不变性被破坏: {d1:.3e}"
    assert d2 < 1e-5, f"边界头前缀不变性被破坏: {d2:.3e}"


# ===========================================================================
# 边界修正
# ===========================================================================
def test_refine_moves_boundary_backward() -> None:
    """
    回归说「迟到了 3 帧」,起点就该往前挪 3 帧。
    """
    segs = [ActionSegment(start_frame=35, end_frame=68, label_id=1, confidence=0.9)]
    # 段首帧说「距起点 3 帧」,段尾前一帧说「距终点 2 帧」
    bd = np.zeros((100, 2))
    bd[35, 0] = 3.0
    bd[67, 1] = 2.0

    out = refine_boundaries_with_regression(segs, bd, fps=30.0)
    assert out[0].start_frame == 32, f"起点应从 35 挪到 32,实际 {out[0].start_frame}"
    assert out[0].end_frame == 70, f"终点应从 68 挪到 70,实际 {out[0].end_frame}"


def test_refine_respects_max_adjust() -> None:
    """回归给出离谱值时,调整量要被 max_adjust 限制住。"""
    segs = [ActionSegment(start_frame=50, end_frame=80, label_id=1, confidence=0.9)]
    bd = np.zeros((100, 2))
    bd[50, 0] = 999.0        # 离谱预测

    out = refine_boundaries_with_regression(segs, bd, fps=30.0, max_adjust=10)
    assert out[0].start_frame == 40, f"起点应只挪 10 帧,实际 {out[0].start_frame}"


def test_refine_does_not_cross_neighbor() -> None:
    """
    相邻段不能被修正到重叠。

    ⚠️ 注意约束是"不重叠",不是"不能越过前一段的终点"。
       往回修正正是边界回归头存在的意义 —— 如果前一段的终点会挡住
       后一段往回挪,那这个功能就完全失效了(实测踩过这个坑)。
    """
    segs = [
        ActionSegment(start_frame=10, end_frame=30, label_id=1, confidence=0.9),
        ActionSegment(start_frame=32, end_frame=50, label_id=2, confidence=0.9),
    ]
    bd = np.zeros((100, 2))
    bd[10, 0] = 20.0    # 想往回挪 20 帧 → 会撞到上一段的范围
    bd[29, 1] = 20.0    # 想往后挪 20 帧 → 会撞到下一段的范围

    out = refine_boundaries_with_regression(segs, bd, fps=30.0, max_adjust=25)
    assert out[0].end_frame <= out[1].start_frame, "相邻段重叠了"
    assert out[1].start_frame >= out[0].end_frame, "相邻段重叠了"


def test_refine_can_move_start_backward_past_previous_end() -> None:
    """
    关键行为:后一段应该能往回修正,越过前一段的**终点**。

    这是边界回归头的主要价值 —— 修正"边界迟到了"的问题。
    如果这个能力被前一段挡住了,整个功能就是废的。
    """
    segs = [
        ActionSegment(start_frame=0, end_frame=32, label_id=0, confidence=0.9),
        ActionSegment(start_frame=32, end_frame=69, label_id=1, confidence=0.9),
    ]
    bd = np.zeros((100, 2))
    bd[32, 0] = 2.0     # 第二段说「我迟到了 2 帧」

    out = refine_boundaries_with_regression(segs, bd, fps=30.0, max_adjust=10)
    assert out[1].start_frame == 30, f"第二段应能往回挪到 30,实际 {out[1].start_frame}"


def test_refine_improves_timestamps_end_to_end() -> None:
    """
    端到端验证:带边界回归比不带更准。

    模拟一个真实场景:Hysteresis 因为"要等置信度涨过阈值"而迟到。

    真值(逐帧标签)
    --------------
        站立  : [0, 30)
        抬手  : [30, 70)      ← 这一段是主角
        站立  : [70, 120)

    「完美」的回归头怎么报
    ----------------------
    回归头给**每一帧**报两个数:
        bd[t, 0] = 这一帧距「它自己所属段」的起点有多远
        bd[t, 1] = 这一帧距「它自己所属段」的终点有多远

    ⚠️ 这两条必须自洽:一帧只能属于一个段。
       所以第 69 帧属于抬手(最后 1 帧),它的
           距终点 = 0,距起点 = 69 - 30 = 39
       而第 70 帧属于站立(第一帧),它的
           距起点 = 0,距终点 = 50
    """
    T, FPS = 120, 30.0
    TRUE_START, TRUE_END = 30, 70

    # 构造概率:动作从第 30 帧开始,但置信度是"慢慢爬升"的
    probs = np.zeros((T, 2))
    for t in range(T):
        if TRUE_START <= t < TRUE_END:
            edge = min(t - TRUE_START, TRUE_END - 1 - t)
            p = 0.45 + 0.5 * min(edge / 10.0, 1.0)
        else:
            p = 0.1
        probs[t, 1] = p
        probs[t, 0] = 1.0 - p

    segs = decode_segments(probs, fps=FPS, high=0.7, low=0.4, min_duration=0.3)
    target = [s for s in segs if s.label_id == 1][0]

    err_before = (
        abs(target.start_frame - TRUE_START),
        abs(target.end_frame - TRUE_END),
    )

    # ── 构造"完美"的回归预测 ────────────────────────────────
    # 逐段构造,保证每帧只属于一段(和 compute_boundary_targets 的语义一致)
    bd = np.zeros((T, 2))

    # 段1:站立 [0, 30)
    for t in range(0, TRUE_START):
        bd[t, 0] = t                       # 距起点
        bd[t, 1] = TRUE_START - t - 1     # 距终点

    # 段2:抬手 [30, 70)
    for t in range(TRUE_START, TRUE_END):
        bd[t, 0] = t - TRUE_START
        bd[t, 1] = TRUE_END - t - 1

    # 段3:站立 [70, 120)
    for t in range(TRUE_END, T):
        bd[t, 0] = t - TRUE_END
        bd[t, 1] = T - t - 1

    refined = refine_boundaries_with_regression(
        segs, bd, fps=FPS, max_adjust=20, confidence=probs
    )
    r = [s for s in refined if s.label_id == 1][0]

    err_after = (
        abs(r.start_frame - TRUE_START),
        abs(r.end_frame - TRUE_END),
    )

    assert err_after[0] <= err_before[0], "回归后起点反而更差了"
    assert err_after[1] <= err_before[1], "回归后终点反而更差了"
    assert err_after == (0, 0), f"完美回归应该完全修正边界,实际误差 {err_after}"
    assert err_after == (0, 0), f"完美回归应该完全修正边界,实际误差 {err_after}"


def test_refine_clamps_to_sequence_bounds() -> None:
    """不能越出序列范围。"""
    segs = [ActionSegment(start_frame=2, end_frame=10, label_id=1, confidence=0.9)]
    bd = np.zeros((12, 2))
    bd[2, 0] = 50.0     # 想往回挪 50 帧 → 应该是 0
    bd[9, 1] = 50.0     # 想往后挪 50 帧 → 应该被 T=12 截断

    out = refine_boundaries_with_regression(segs, bd, fps=30.0, max_adjust=100)
    assert out[0].start_frame >= 0, "起点越界"
    assert out[0].end_frame <= 12, "终点越界"


def test_refine_updates_confidence() -> None:
    """给定概率表时,置信度要用修正后区间的平均值重算。"""
    segs = [ActionSegment(start_frame=10, end_frame=20, label_id=1, confidence=0.5)]
    bd = np.zeros((30, 2))
    probs = np.zeros((30, 3))
    probs[:, 1] = 0.8

    out = refine_boundaries_with_regression(segs, bd, fps=30.0, confidence=probs)
    assert abs(out[0].confidence - 0.8) < 1e-6, f"置信度应重算为 0.8,实际 {out[0].confidence}"


def test_refine_empty_segments() -> None:
    """空输入不报错。"""
    out = refine_boundaries_with_regression([], np.zeros((10, 2)))
    assert out == []


def test_refine_zero_prediction_changes_nothing() -> None:
    """回归预测全 0 时,边界保持原样。"""
    segs = [ActionSegment(start_frame=10, end_frame=20, label_id=1, confidence=0.9)]
    out = refine_boundaries_with_regression(segs, np.zeros((30, 2)))
    assert out[0].start_frame == 10 and out[0].end_frame == 20


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
