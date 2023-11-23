"""
ActionFlow 的模型层。

三层架构里的 Layer 2(动作分类)在这一层。

四种架构
--------
    offline   : 整段视频打一个标签
    streaming : 因果流式,逐帧输出(前缀不变)★ 默认
    boundary  : 流式 + 边界回归头
    fusion    : 骨架 + 视频双路

统一用 ``build_model_from_config`` 构建,不要直接 new 某个类 ——
这样全项目只有一处知道"哪种配置对应哪种模型"。
"""

from actionflow.models.boundary import (
    BoundaryAwareClassifier,
    BoundaryRegressionHead,
    boundary_regression_loss,
    build_boundary_aware_model,
    compute_boundary_targets,
)
from actionflow.models.classifier import (
    SkeletonEmbedding,
    SkeletonTransformerClassifier,
    build_model,
)
from actionflow.models.factory import (
    ARCH_BOUNDARY,
    ARCH_FUSION,
    ARCH_OFFLINE,
    ARCH_STREAMING,
    VALID_ARCHS,
    arch_produces_boundary,
    arch_supports_sequence_output,
    build_model_from_config,
)
from actionflow.models.fusion import (
    CrossAttentionFusion,
    FusionClassifier,
    SimpleVideoBackbone,
    VideoBackbone,
)
from actionflow.models.streaming import (
    CausalStreamingClassifier,
    build_causal_frame_mask,
    build_streaming_model,
    export_streaming_onnx,
)

__all__ = [
    # 分类器
    "SkeletonEmbedding",
    "SkeletonTransformerClassifier",
    "build_model",
    # 流式
    "CausalStreamingClassifier",
    "build_causal_frame_mask",
    "build_streaming_model",
    "export_streaming_onnx",
    # 边界
    "BoundaryAwareClassifier",
    "BoundaryRegressionHead",
    "boundary_regression_loss",
    "build_boundary_aware_model",
    "compute_boundary_targets",
    # 融合
    "CrossAttentionFusion",
    "FusionClassifier",
    "SimpleVideoBackbone",
    "VideoBackbone",
    # 工厂
    "build_model_from_config",
    "arch_supports_sequence_output",
    "arch_produces_boundary",
    "ARCH_OFFLINE",
    "ARCH_STREAMING",
    "ARCH_BOUNDARY",
    "ARCH_FUSION",
    "VALID_ARCHS",
]
