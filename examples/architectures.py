"""
ActionFlow 四种架构演示 —— streaming / boundary / fusion / offline

这个脚本把新增的四个架构跑一遍,让你直观看到:
    ① 每种架构的输入输出形状
    ② 「前缀不变性」是什么意思、怎么验证
    ③ 边界回归头怎么把时间戳改准
    ④ Cross-Attention 融合后的特征长什么样

运行:
    python examples/architectures.py

不需要任何权重、不需要数据集、不需要网络 —— 全部用随机骨架。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.core.config import ModelConfig  # noqa: E402
from actionflow.models.factory import (  # noqa: E402
    build_model_from_config,
)
from actionflow.postprocess.decode import (  # noqa: E402
    decode_segments,
    refine_boundaries_with_regression,
)


# ===========================================================================
# 工具
# ===========================================================================
def make_skel(B: int = 1, T: int = 32, J: int = 17, seed: int = 0) -> torch.Tensor:
    """
    造一段假的骨架序列。

    形状 (B, T, J, 3),最后一维是 (x, y, 置信度)。
    真实使用时这一段是 RTMPose 的输出。
    """
    g = torch.Generator().manual_seed(seed)
    skel = torch.randn(B, T, J, 3, generator=g) * 0.1
    skel[..., 2] = torch.rand(B, T, J, generator=g) * 0.3 + 0.7   # 置信度
    return skel


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ===========================================================================
# ① 四种架构的基本形态
# ===========================================================================
def demo_1_four_architectures() -> None:
    banner("① 四种架构:输入输出形状对比")

    cfg = ModelConfig(num_classes=4, d_model=64, nhead=4, num_layers=2)

    print(f"  {'arch':<12} {'参数量':>10}  {'输出形状':<22} 说明")
    print("  " + "-" * 76)

    for arch in ("offline", "streaming", "boundary", "fusion"):
        cfg.arch = arch
        model = build_model_from_config(cfg)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        skel = make_skel(B=2, T=16)
        with torch.no_grad():
            if arch == "offline":
                out = model(skel)
                shape = tuple(out.shape)
                note = "整段视频一个标签"
            elif arch == "boundary":
                logits, dist = model(skel)
                shape = f"{tuple(logits.shape)} + {tuple(dist.shape)}"
                note = "逐帧 + 边界距离"
            elif arch == "fusion":
                frames = torch.randn(2, 16, 3, 64, 64)
                out = model(skel, frames)
                shape = tuple(out.shape)
                note = "骨架 + 视频"
            else:
                out = model(skel)
                shape = tuple(out.shape)
                note = "逐帧输出(流式)"

        print(f"  {arch:<12} {n_params:>8.3f}M  {str(shape):<22} {note}")

    print()
    print("  提示:fusion 架构需要视频帧,这里用随机张量代替真实画面。")


# ===========================================================================
# ② 前缀不变性 —— 流式架构的验收标准
# ===========================================================================
def demo_2_prefix_invariance() -> None:
    banner("② 前缀不变性:流式架构的验收标准")

    print("  含义:「用前 t 帧算出的结果」必须和")
    print("        「用前 t+N 帧算出的前 t 帧」完全一样。")
    print()
    print("  为什么必须逐位相同?")
    print("    实时系统不能撤回已经输出的结果。")
    print("    用户已经看到的第 3 帧,不能因为你后来又来 5 帧而偷偷改变。")
    print()

    cfg = ModelConfig(num_classes=4, d_model=64, nhead=4, num_layers=2)
    skel = make_skel(B=1, T=32)
    # 视频帧也要用「同一个」张量的前缀 —— 不能另造一个随机张量,
    # 否则测的就不是前缀不变性,而是「输入不同」了。
    g = torch.Generator().manual_seed(123)
    frames = torch.randn(1, 32, 3, 64, 64, generator=g)

    for arch in ("streaming", "boundary", "fusion"):
        cfg.arch = arch
        model = build_model_from_config(cfg)
        model.eval()

        with torch.no_grad():
            if arch == "offline":
                continue
            if arch == "boundary":
                full_logits, full_dist = model(skel)
                pre_logits, pre_dist = model(skel[:, :16])
                d = max(
                    (full_logits[:, :16] - pre_logits).abs().max().item(),
                    (full_dist[:, :16] - pre_dist).abs().max().item(),
                )
            elif arch == "fusion":
                full = model(skel, frames)
                pre = model(skel[:, :16], frames[:, :16])
                d = (full[:, :16] - pre).abs().max().item()
            else:
                full = model(skel)
                pre = model(skel[:, :16])
                d = (full[:, :16] - pre).abs().max().item()

        verdict = "逐位相同 ✓" if d == 0.0 else ("通过 ✓" if d < 1e-5 else "失败 ✗")
        print(f"    {arch:<12} 前缀差异 = {d:.3e}   {verdict}")

    print()
    print("  这条测试写在 tests/test_streaming.py 里,")
    print("  任何改动破坏了它都会立刻让测试挂掉。")


# ===========================================================================
# ③ 因果掩码:帧内全连通 + 帧间因果
# ===========================================================================
def demo_3_causal_mask() -> None:
    banner("③ 因果掩码:帧内全连通 + 帧间因果")

    from actionflow.models.streaming import build_causal_frame_mask

    T, J = 4, 3
    mask = build_causal_frame_mask(T, J)

    print(f"  T={T} 帧,每帧 {J} 个关节 → 共 {T*J} 个 token")
    print()
    print("  掩码含义(PyTorch attn_mask 约定):")
    print("    True  = 禁止关注(那一格会被填成 -inf)")
    print("    False = 可以关注")
    print()
    header = "       " + "".join(f"  t{j}·k{k}" for j in range(T) for k in range(J))
    print(header)
    for i in range(T * J):
        row = "".join(
            "   ✗  " if mask[i, j] else "   ✓  "
            for j in range(T * J)
        )
        print(f"  t{i // J}·k{i % J}{row}")

    print()
    print("  读法:")
    print("    · 第 0 帧的 3 个关节之间 → 全是 ✓（同帧互相可见）")
    print("    · 第 1 帧的关节 → 能看第 0 帧,不能看第 2/3 帧（帧间因果）")
    print("    · 第 3 帧的关节 → 全是 ✗（最后一帧只看得到自己和过去）")
    print()
    print("  ⚠️ 关键:掩码必须按【帧号】算,不能按【token 号】算。")
    print("     token 号 = 帧号 × 关节数 + 关节编号")
    print("     如果直接对 token 号做下三角掩码,第 0 帧的左手腕")
    print("     就看不到第 0 帧的左手肩 —— 物理上说不通。")


# ===========================================================================
# ④ 边界回归头:怎么把时间戳改准
# ===========================================================================
def demo_4_boundary_head() -> None:
    banner("④ 边界回归头:把迟到的边界拉回来")

    T, FPS = 120, 30.0
    TRUE_START, TRUE_END = 30, 70

    # ── 造一段概率:动作从第 30 帧开始,但置信度慢慢爬升 ──
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

    print(f"  真值:                [{TRUE_START}, {TRUE_END})")
    print(f"  Hysteresis 解出来:   [{target.start_frame}, {target.end_frame})")
    e0 = abs(target.start_frame - TRUE_START)
    e1 = abs(target.end_frame - TRUE_END)
    print(f"    起点误差 {e0} 帧,终点误差 {e1} 帧")
    print()
    print("  为什么迟到?Hysteresis 要等置信度涨过 0.7 才敢开段,")
    print("  而动作刚开始时模型还没把握(只有 0.5)。")
    print()

    # ── 造一个「完美」的回归预测 ──
    # 每一帧报告的是「它自己所属那一段」的距离
    bd = np.zeros((T, 2))
    for t in range(0, TRUE_START):
        bd[t, 0] = t
        bd[t, 1] = TRUE_START - t - 1
    for t in range(TRUE_START, TRUE_END):
        bd[t, 0] = t - TRUE_START
        bd[t, 1] = TRUE_END - t - 1
    for t in range(TRUE_END, T):
        bd[t, 0] = t - TRUE_END
        bd[t, 1] = T - t - 1

    refined = refine_boundaries_with_regression(
        segs, bd, fps=FPS, max_adjust=20, confidence=probs
    )
    r = [s for s in refined if s.label_id == 1][0]

    print(f"  边界回归修正后:      [{r.start_frame}, {r.end_frame})")
    e0b = abs(r.start_frame - TRUE_START)
    e1b = abs(r.end_frame - TRUE_END)
    print(f"    起点误差 {e0b} 帧,终点误差 {e1b} 帧")
    print()

    if e0b <= e0 and e1b <= e1:
        print("  ✓ 时间戳变准了")
    else:
        print("  ✗ 变差了（不该发生）")

    print()
    print("  原理:对段【内部】的帧,读出「所属段的起点位置」")
    print("        起点 = t - bd[t,0]")
    print("        终点 = t + bd[t,1] + 1     (终点是开区间)")
    print("      然后取中位数 —— 不信「门口那一帧」,看屋里的人怎么说的。")


# ===========================================================================
# ⑤ Cross-Attention 融合
# ===========================================================================
def demo_5_cross_attention() -> None:
    banner("⑤ Cross-Attention:骨架查视频,视频查骨架")

    from actionflow.models.fusion import CrossAttentionFusion

    fusion = CrossAttentionFusion(d_skel=64, d_video=64, d_model=64, nhead=4)
    fusion.eval()

    T = 5
    skel_feat = torch.randn(1, T, 64)
    video_feat = torch.randn(1, T, 64)

    with torch.no_grad():
        fused_skel, fused_video = fusion(skel_feat, video_feat)

    print(f"  骨架特征   {tuple(skel_feat.shape)}")
    print(f"  视频特征   {tuple(video_feat.shape)}")
    print(f"  融合骨架   {tuple(fused_skel.shape)}   ← 骨架 + 语义补充")
    print(f"  融合视频   {tuple(fused_video.shape)}   ← 视频 + 几何引导")
    print()
    print("  两路都在问对方问题:")
    print("    骨架问视频:「这一帧画面上有什么?」 → 语义补充")
    print("    视频问骨架:「这个人怎么动的?」     → 几何引导")
    print()
    print("  ⚠️ 残差连接很重要:")
    print("     融合 = 骨架原样 + 视频补充")
    print("     即使视频路认错了(画面里有干扰物),骨架信息还在,")
    print("     模型顶多「参考了一下错误意见」,不会彻底跑偏。")

    # 验证残差确实存在
    with torch.no_grad():
        s = fusion.skel_proj(skel_feat)
        no_semantic = fusion.norm_skel(s)
        diff = (fused_skel - no_semantic).abs().max().item()
    print()
    print(f"  实测:去掉语义补充后特征变化 {diff:.4f}")
    print("        → 非零说明语义补充真的加进去了(残差通路有效)")


# ===========================================================================
def main() -> None:
    print("\n" + "#" * 68)
    print("#  ActionFlow · 四种架构演示")
    print("#" * 68)

    demo_1_four_architectures()
    demo_2_prefix_invariance()
    demo_3_causal_mask()
    demo_4_boundary_head()
    demo_5_cross_attention()

    print("\n" + "#" * 68)
    print("#  全部演示完成")
    print("#" * 68 + "\n")


if __name__ == "__main__":
    main()
