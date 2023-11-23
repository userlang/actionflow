"""
模型工厂 —— 根据配置构建对应的模型
==================================

为什么要一个工厂
----------------
现在有四种模型架构,各有各的用途:

    ┌─────────────────────────────────────────────────────────────┐
    │ arch          │ 类                        │ 用途              │
    ├─────────────────────────────────────────────────────────────┤
    │ "offline"     │ SkeletonTransformer...    │ 整段视频打一个标签 │
    │ "streaming"   │ CausalStreaming...        │ 流式,逐帧输出 ★默认│
    │ "boundary"    │ BoundaryAware...          │ 流式 + 精确时间戳  │
    │ "fusion"      │ FusionClassifier          │ 骨架 + 视频双路    │
    └─────────────────────────────────────────────────────────────┘

如果不统一,每个地方都要写一堆 if/else 判断该 new 哪个类,
迟早会漏掉某个分支(而且各处的判断逻辑可能不一致)。

有了工厂,全项目只有**一个地方**知道"哪种配置对应哪种模型"。

一个更重要的理由
----------------
**不同架构的模型,输入输出的"形状契约"是不同的:**

    offline   : (B, T, J, 3) → (B, C)          整段一个结果
    streaming : (B, T, J, 3) → (B, T, C)        每帧一个结果
    boundary  : (B, T, J, 3) → (B, T, C) + (B, T, 2)
    fusion    : (B, T, J, 3) + 视频 → (B, T, C)

调用方要是不小心把 streaming 模型当成 offline 用,
拿到的 (B, T, C) 当成 (B, C) 用,bug 会非常隐蔽
(张量维度对得上,但语义完全错了)。

工厂在构建时就把这些约定统一了,并且在
``supports_sequence_output`` 这类属性里明确标出来。
"""
from __future__ import annotations

ARCH_OFFLINE = "offline"
ARCH_STREAMING = "streaming"
ARCH_BOUNDARY = "boundary"
ARCH_FUSION = "fusion"

VALID_ARCHS = (ARCH_OFFLINE, ARCH_STREAMING, ARCH_BOUNDARY, ARCH_FUSION)


def build_model_from_config(cfg):
    """
    根据 ModelConfig 构建模型。

    Parameters
    ----------
    cfg : ModelConfig

    Returns
    -------
    nn.Module

    Notes
    -----
    不同架构的 forward 签名:
        offline  : forward(skel)                → (B, C)
        streaming: forward_probs(skel)          → (B, T, C)
        boundary : forward_probs(skel)          → ((B, T, C), (B, T, 2))
        fusion   : forward(skel, frames)        → (B, T, C)
    """
    arch = getattr(cfg, "arch", ARCH_STREAMING)

    if arch == ARCH_OFFLINE:
        from actionflow.models.classifier import build_model

        return build_model(cfg)

    if arch == ARCH_STREAMING:
        from actionflow.models.streaming import build_streaming_model

        return build_streaming_model(cfg)

    if arch == ARCH_BOUNDARY:
        from actionflow.models.boundary import build_boundary_aware_model

        return build_boundary_aware_model(cfg)

    if arch == ARCH_FUSION:
        from actionflow.models.fusion import FusionClassifier, SimpleVideoBackbone
        from actionflow.models.streaming import build_streaming_model

        base = build_streaming_model(cfg)
        backbone = SimpleVideoBackbone(out_dim=cfg.d_model)
        return FusionClassifier(
            base,
            backbone,
            d_video=cfg.d_model,
            nhead=cfg.nhead,
            dropout=cfg.dropout,
        )

    raise ValueError(
        f"未知的 arch='{arch}'，可选: {', '.join(VALID_ARCHS)}"
    )


def arch_supports_sequence_output(arch: str) -> bool:
    """
    这个架构是不是逐帧输出的?

    offline 架构是整段一个结果,不逐帧输出 ——
    流式管线不能用它。
    """
    return arch != ARCH_OFFLINE


def arch_produces_boundary(arch: str) -> bool:
    """这个架构会不会输出边界回归的预测?"""
    return arch == ARCH_BOUNDARY
