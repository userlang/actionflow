"""
Skeleton Action Classifier — 骨架时序 Transformer 分类器
=======================================================

Layer 2 · Action Recognition 的核心模型。

结构
----
    输入  (B, T, J, 3)   骨架序列：T 帧 × J 关节 × (x, y, conf)
      ↓
    [投影]  (B, T, J, D)  每个关节的 (x,y,conf) 投到 d_model 维
      ↓
    [关节嵌入]  + nn.Embedding(J, D)      告诉模型"这是左手腕还是右脚踝"
      ↓
    [时间嵌入]  + nn.Embedding(T, D)      告诉模型"这是第几帧"
      ↓
    [展平]  (B, T*J, D)   整段视频变成 T*J 个 token
      ↓
    [TransformerEncoder]  (B, T*J, D)   自注意力建模任意两个关节/时刻的关系
      ↓
    [池化]  (B, D)        时间维 max-pool + 关节维 mean-pool
      ↓
    [分类头]  (B, C)      动作类别 logits

关键设计
--------
1. **Key-Padding Mask**：关键点置信度低于阈值视为漏检，该 token 被忽略。
   这是骨架任务必备——漏检的关节如果参与注意力会污染整个序列。

2. **因果注意力掩码**（可选）：`causal=True` 时给 TransformerEncoder 传入
   三角 mask，使每个时刻只能看到历史帧，满足流式在线推理的要求。

3. **不使用 nn.TransformerEncoder 的默认行为**：显式传入 mask 而非依赖
   默认的 nested tensor 优化，避免 ONNX 导出时出问题。
   （`enable_nested_tensor=False` 是显式声明，不是可选项。）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SkeletonEmbedding(nn.Module):
    """关节嵌入 + 时间嵌入 + 坐标投影。"""

    def __init__(self, num_joints: int, d_model: int, max_window: int = 256) -> None:
        super().__init__()
        # (x, y, conf) → d_model
        self.coord_proj = nn.Linear(3, d_model)
        # 关节身份嵌入
        self.joint_embed = nn.Embedding(num_joints, d_model)
        # 时间位置嵌入
        self.time_embed = nn.Embedding(max_window, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        skel: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            skel:      (B, T, J, 3)
            joint_ids: (J,) 关节 id，默认 0..J-1
        Returns:
            (B, T*J, D)
        """
        B, T, J, C = skel.shape
        D = self.coord_proj.out_features

        if joint_ids is None:
            joint_ids = torch.arange(J, device=skel.device)

        # (B, T, J, D)
        x = self.coord_proj(skel)

        # 关节嵌入：(J, D) → (1, 1, J, D) 广播
        je = self.joint_embed(joint_ids).view(1, 1, J, D)
        x = x + je

        # 时间嵌入：(T, D) → (1, T, 1, D) 广播
        te = self.time_embed(torch.arange(T, device=skel.device)).view(1, T, 1, D)
        x = x + te

        x = self.dropout(x)

        # 展平 (B, T, J, D) → (B, T*J, D)
        return x.reshape(B, T * J, D)


class SkeletonTransformerClassifier(nn.Module):
    """
    骨架时序 Transformer 动作分类器。

    Args:
        num_classes:  动作类别数
        num_joints:   关节数（COCO = 17）
        d_model:      token 维度
        nhead:        注意力头数（必须整除 d_model）
        num_layers:   Transformer 层数
        dim_feedforward: FFN 中间维度
        dropout:      dropout 比例
        window:       训练时的窗口长度 T
        causal:       是否使用因果掩码（流式）
    """

    def __init__(
        self,
        num_classes: int = 60,
        num_joints: int = 17,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        window: int = 32,
        causal: bool = False,
        conf_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.d_model = d_model
        self.window = window
        self.causal = causal
        self.conf_threshold = conf_threshold

        self.embedding = SkeletonEmbedding(num_joints, d_model, max_window=max(window * 2, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # enable_nested_tensor=False: 显式传入 mask 时走 dense 路径，
        # 既避免 nested tensor 的 prototype 警告，也保证 ONNX 导出图干净。
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 池化后的两层 MLP 分类头（输入 2*d_model）
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # ------------------------------------------------------------------
    def build_masks(
        self,
        skel: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        构造两种 mask。

        注意 PyTorch 的约束：当同时传入 mask 和 src_key_padding_mask 时，
        两者必须"类型一致"。src_key_padding_mask 是 bool 类型，
        所以 attn_mask 也用 bool（True = 禁止关注）。

        Returns:
            attn_mask:         (T*J, T*J) bool 因果注意力 mask，或 None
            key_padding_mask:  (B, T*J) bool padding mask
        """
        B, T, J, _ = skel.shape
        device = skel.device

        # ── Key-Padding Mask：置信度低的关节被忽略 ──────────────────
        conf = skel[..., 2]                       # (B, T, J)
        valid = conf > self.conf_threshold        # (B, T, J)

        # 防 NaN：如果某个样本的所有 token 都被 mask 掉，注意力 softmax 会
        # 对全 -inf 求值而溢出。这里保证每个样本至少有一个有效 token。
        #
        # 注意：不能用 `if all_masked.any():` 这种数据依赖分支 ——
        # torch.onnx.export (torch.export) 无法 trace 它。
        # 用无分支的写法：构造一个 one-hot 后与 all_masked 做外积。
        all_masked = ~valid.any(dim=-1)           # (B, T) 该帧是否全无效
        first_joint = torch.zeros_like(valid)
        first_joint[:, :, 0] = 1.0                # (B, T, J) 只有第0个关节为1
        valid = valid | (all_masked.unsqueeze(-1).to(valid.dtype) * first_joint)

        key_padding_mask = ~valid.reshape(B, T * J)  # True = 忽略

        # ── 因果注意力 Mask：只允许看"当前帧及更早的帧" ────────────
        # ⚠️ 关键点：掩码必须按**帧号**判断，不能按 token 编号。
        #    展平顺序是 token = 帧号*J + 关节号，所以简单用上三角
        #    （禁止看编号更大的 token）会连**同一帧内的关节**也隔开 ——
        #    左手腕就看不到左手肩了，判断"抬手"必然出错。
        #
        #    正确规则：允许看 ⟺ 对方的帧号 ≤ 自己的帧号
        #    （帧内全连通 + 帧间因果）
        #
        # 用 bool 而不是 -inf float，保持与 key_padding_mask 类型一致。
        attn_mask = None
        if self.causal:
            from actionflow.models.streaming import build_causal_frame_mask

            attn_mask = build_causal_frame_mask(T, J, device=device)

        return attn_mask, key_padding_mask

    # ------------------------------------------------------------------
    def forward(self, skel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skel: (B, T, J, 3) 骨架序列
        Returns:
            (B, num_classes) 分类 logits
        """
        x = self.embedding(skel)                       # (B, T*J, D)
        attn_mask, pad_mask = self.build_masks(skel)

        out = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=pad_mask,
        )                                               # (B, T*J, D)

        # 池化：时间维取 max（抓最显著动作），关节维取 mean（整体姿态）
        # 注意维度：out 是 (B, T, J, D)
        #   max over 时间 → (B, J, D) → 再对关节取 mean → (B, D)
        #   mean over 关节 → (B, T, D) → 再对时间取 mean → (B, D)
        B, TJ, D = out.shape
        T, J = skel.shape[1], skel.shape[2]
        out = out.view(B, T, J, D)

        pooled_t = out.max(dim=1).values.mean(dim=1)      # (B, D)  时间max + 关节mean
        pooled_j = out.mean(dim=2).max(dim=1).values       # (B, D)  关节mean + 时间max

        pooled = torch.cat([pooled_t, pooled_j], dim=-1)   # (B, 2D)

        return self.classifier_head(pooled)

    # ------------------------------------------------------------------
    def forward_sequence(self, skel: torch.Tensor) -> torch.Tensor:
        """
        逐帧输出版本 —— 对每个前缀单独跑一次完整分类器。

        对窗口内的每个时间点 t，输出「截至第 t 帧」的分类概率，
        得到 (B, T, C)。

        ⚠️ 这是**朴素实现**，计算量 O(T²)（要跑 T 次完整前向）。
           它存在的意义是当**正确性参照**（reference implementation），
           用来验证 `CausalStreamingClassifier` 的结果是否一致。

           真实的流式推理请用 ``actionflow.models.streaming.CausalStreamingClassifier``：
           那里有正确的因果掩码，一次前向就能算出全部 T 帧的结果。

        为什么朴素实现一定正确？
        因为它对每个前缀 t 都**完整地**重新算了一遍，
        第 t 帧的输出只可能用到前 t 帧的信息，
        所以天然满足前缀不变性 —— 只是慢。

        Args:
            skel: (B, T, J, 3)
        Returns:
            (B, T, C) 逐帧概率（softmax 后）
        """
        B, T, J, _ = skel.shape
        outs = []
        for t in range(1, T + 1):
            logits = self.forward(skel[:, :t])       # (B, C)
            outs.append(logits)
        stacked = torch.stack(outs, dim=1)           # (B, T, C)
        return stacked.softmax(dim=-1)


def build_model(cfg) -> SkeletonTransformerClassifier:
    """从 ModelConfig 构建模型。"""
    return SkeletonTransformerClassifier(
        num_classes=cfg.num_classes,
        num_joints=cfg.num_joints,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        window=cfg.window,
        causal=cfg.causal,
    )
