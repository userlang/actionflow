"""
双路融合模型:骨架路 + 视频路
=============================

为什么要两条路
------------
现在的 ActionFlow 只有**骨架路**:RTMPose 抽出 17 个关键点,Transformer 分析关键点的运动。

骨架路的优点:
    - 极其轻量(0.3M 参数)
    - 天然抗外观变化(穿什么衣服、在什么背景下都不影响)
    - 端侧友好

骨架路的**致命缺点**:
    - **关键点描述不了的信息,骨架路就看不见。**

    举几个例子:
      ① 「拿起杯子」和「拿起手机」—— 手的位置一样,但手里有东西 vs 手里是空的,
         骨架完全一样,模型分不出来。
      ② 「开门」和「挥手」—— 手的轨迹可能相似,但门把手的存在与否不同。
      ③ 「走路」和「跑步」—— 关节角度可能接近,但步频、步幅、身体摆动幅度不同,
         这些信息在关节坐标里体现得很微弱。

    说白了:骨架路看到的是**几何**,看不到**语义**。

视频路能补上这一块
------------------
视频路直接看画面像素,颜色、纹理、物体、有没有道具,全都在里面。

代价是:视频路很重。Video Swin-T 有 2800 万参数,MobileNet 那类的轻量模型也有几百万,
而且计算量比骨架路高两三个数量级。

所以本项目的立场是
----------------
**骨架路是主力,视频路是可选增强。**

    默认配置(arch="streaming"):只跑骨架路,0.3M 参数,端侧可跑
    增强配置(arch="fusion"):骨架 + 视频,精度更高,但需要 GPU

两种融合方式
------------
方式 A:特征拼接(简单,但效果一般)
    把视频特征和骨架特征在**每一帧**上拼起来,一起送进 Transformer。

    问题:视频特征和骨架特征**尺度完全不同**。
         视频特征可能是 [-3, 3] 的深层特征,骨架是 [-2, 2] 的坐标,
         直接拼会让训练很难受(梯度都被大尺度的那个主导)。

方式 B:Cross-Attention 交叉注意力(本文件采用)
    让两条路**互相看**对方,但各自保留自己的"主赛道"。

    通俗讲:
        骨架路问视频路:「这一帧画面上有什么?」 → 得到"语义补充"
        视频路问骨架路:「这个人的姿态怎么动的?」 → 得到"几何引导"

    这样做的好处:
        - 两条路的信息**没有混在一起**(不像拼接那样),各自的任务保持纯粹
        - 视频路的信息是"通过注意力加权"进来的,权重由模型自己学,
          相当于模型自己决定「这一帧该不该多看画面一眼」
        - 视频路是**可选**的:关掉它,骨架路照常工作

Cross-Attention 是什么
----------------------
用"查字典"来理解:

    骨架路的特征 = 查询词 (Query)
    视频路的特征 = 字典 (Key / Value)

    「用骨架路当查询词,去视频路这个字典里查」:
    每个骨架 token 问:「和我最相关的画面信息是什么?」
    注意力机制算出相关度,加权求和 → 得到"这一帧的语义补充"。

    反过来,视频路当查询词、骨架路当字典,得到"几何引导"。

实现
----
    骨架特征 (B, T, D_s)  ──┐
                             ├─→ CrossAttention(查询=骨架, 字典=视频) → (B, T, D_v)
    视频特征 (B, T, D_v)  ──┘
                             │
                             ↓
    融合 = 骨架 + 投影(语义补充)     ← 残差连接,原始骨架信息不丢
    再过一遍 Transformer → 分类头

为什么用残差连接
----------------
如果融合后的特征 = 纯视频特征,那么"这一帧在做什么动作"的信息(骨架)
就被冲掉了。

残差连接的做法是:融合结果 = 骨架原样 + 视频补充信息。

    原始骨架 + 一点语义补充

这样即使视频路认错了(比如画面里有干扰物),骨架信息还在,
模型顶多"参考了一下错误意见",不会彻底跑偏。

这是深度学习里非常常用、非常有效的技巧 —— 残差连接让"多加信息"
变成"锦上添花"而不是"喧宾夺主"。

视频 backbone 怎么来
-------------------
本项目**不训练**视频 backbone(那需要 ImageNet 预训练权重和大量算力),
而是提供一个**接口**,让用户自己接:

    - torchvision 的 video 模型(Video Swin-T / MViT / R(2+1)D)
    - MMPose 的 video backbone
    - 任何输出 (B, T, D_v) 的时序特征提取器

接口是 ``VideoBackbone`` 抽象基类,实现 ``forward(frames) -> (B, T, D_v)`` 即可。

这样项目保持轻量,同时又留出了扩展口子 —— 想接双路融合的人有明确的
接入方式,又不会因为塞进来一个 2800 万参数的模型而让端侧部署的承诺破产。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VideoBackbone(nn.Module):
    """
    视频 backbone 的抽象接口。

    子类需要实现 ``forward``,把一段视频变成一串时序特征。

    约定
    ----
    输入:
        frames: (B, T, 3, H, W)
            B = 批次, T = 帧数, 3 = RGB 通道, H×W = 空间尺寸

    输出:
        (B, T, D_v)
            每帧一个 D_v 维的特征向量

    怎么实现一个具体的 backbone
    --------------------------
    思路是把视频 backbone 当成「图像 backbone + 时间维聚合」:

        ① 逐帧过图像 backbone → (B*T, D)
        ② reshape 回 (B, T, D)
        ③ 时间维做个池化或轻量时序模块 → (B, T, D)

    例如用 torchvision 的 swin_t:
        from torchvision.models.video import swin3d_t
        net = swin3d_t(weights="DEFAULT")
        # 替换最后的分类头,取倒数第二层特征
    """

    def forward(self, frames: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("子类必须实现 forward")


class SimpleVideoBackbone(VideoBackbone):
    """
    轻量视频 backbone —— 用可训练的小卷积从像素里抽特征。

    ⚠️ 这个实现**不是**用来刷 SOTA 的,它的定位是:
        1. 让整条双路链路**能跑通、能测试**(不依赖下载预训练权重)
        2. 给"我想接自己的 backbone"提供一个具体的参照实现

    真实项目里应该换成 Video Swin-T / MViT 之类的预训练模型。

    结构
    ----
        (B, T, 3, H, W)
          ↓ reshape 成 (B*T, 3, H, W) 当作一批图片
          ↓ 3D 卷积 ×2 (在时间维也卷,所以能捕捉运动)
          ↓ 全局池化
        (B*T, D_v)
          ↓ reshape 回 (B, T, D_v)

    为什么用 3D 卷积而不是 2D?
    ------------------
    2D 卷积只看单帧,分不清"手往上抬"和"手往下放"——
    因为这两种情况每一帧看起来一样(只是顺序不同)。
    3D 卷积的卷积核在时间维也有厚度,能看出"帧与帧之间怎么变的"。

    通俗讲:2D 像看一堆单张照片,3D 像看一段小视频。
    """

    def __init__(self, out_dim: int = 128, num_classes: int = 0) -> None:
        super().__init__()
        self.out_dim = out_dim

        self.net = nn.Sequential(
            # (B*T, 3, H, W) → (B*T, 32, H/2, W/2)
            nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # → (B*T, 64, H/4, W/4)
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # → (B*T, out_dim, H/8, W/8)
            nn.Conv3d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),      # → (B*T, out_dim, 1, 1, 1)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W)
        Returns:
            (B, T, out_dim)
        """
        B, T, C, H, W = frames.shape

        # (B, T, 3, H, W) → (B*T, 3, H, W)
        x = frames.reshape(B * T, C, H, W)

        # 3D 卷积需要时间维,补一个长度为 1 的时间维 → (B*T, 3, 1, H, W)
        x = x.unsqueeze(2)
        x = self.net(x)                       # (B*T, out_dim, 1, 1, 1)
        x = x.flatten(1)                      # (B*T, out_dim)

        # → (B, T, out_dim)
        return x.reshape(B, T, self.out_dim)


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块。

    做两件事:
        ① 骨架路(查询) 去视频路(字典)里查 → 得到语义补充
        ② 视频路(查询) 去骨架路(字典)里查 → 得到几何引导

    然后用残差连接把补充信息加回各自的路上。

    参数
    ----
    d_skel: int
        骨架特征维度
    d_video: int
        视频特征维度
    d_model: int
        融合后的统一维度
    nhead: int
        注意力头数
    """

    def __init__(
        self,
        d_skel: int = 128,
        d_video: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # 维度对齐:两条路的维度可能不同,先各自投影到 d_model
        self.skel_proj = nn.Linear(d_skel, d_model)
        self.video_proj = nn.Linear(d_video, d_model)

        # 交叉注意力的查询/键/值投影
        self.q_skel = nn.Linear(d_model, d_model)
        self.k_video = nn.Linear(d_model, d_model)
        self.v_video = nn.Linear(d_model, d_model)

        self.q_video = nn.Linear(d_model, d_model)
        self.k_skel = nn.Linear(d_model, d_model)
        self.v_skel = nn.Linear(d_model, d_model)

        self.out_skel = nn.Linear(d_model, d_model)
        self.out_video = nn.Linear(d_model, d_model)

        self.norm_skel = nn.LayerNorm(d_model)
        self.norm_video = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead

    # ------------------------------------------------------------------
    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        nhead: int,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        标准的多头注意力。

        通俗讲:
            1. 算"谁和谁相关"(query 和 key 做点积)
            2. 把这些相关度变成权重(softmax,加起来等于 1)
            3. 按权重把 value 加权求和

        Parameters
        ----------
        attn_mask: (Lq, Lk) 的 bool 张量,可选
            True = 禁止关注(PyTorch 的 attn_mask 约定)
            传因果掩码时,第 i 个 query 只能看 j ≤ i 的 key。
        """
        B, Lq, D = query.shape
        Lk = key.shape[1]
        head_dim = D // nhead

        # 拆成多头: (B, L, D) → (B, nhead, L, head_dim)
        q = query.view(B, Lq, nhead, head_dim).transpose(1, 2)
        k = key.view(B, Lk, nhead, head_dim).transpose(1, 2)
        v = value.view(B, Lk, nhead, head_dim).transpose(1, 2)

        # 相关度 = q·k,除以 sqrt(head_dim) 做数值缩放
        # (为什么要除?因为维度越高,点积的数值可能越大,
        #  塞进 softmax 会太极端 —— 几乎变成 one-hot,梯度就没了)
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale    # (B, nhead, Lq, Lk)

        # ── 因果掩码 ────────────────────────────────────────
        # ⚠️ 这一行不能省。
        # 注意力默认是**双向**的:第 3 帧会去看第 8 帧的 key。
        # 那就破坏了前缀不变性 —— 实时系统里,
        # 用户已经看到的第 3 帧会因为后来到达的帧而改变。
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = attn.softmax(dim=-1)

        out = attn @ v                                # (B, nhead, Lq, head_dim)
        return out.transpose(1, 2).reshape(B, Lq, D)

    # ------------------------------------------------------------------
    def forward(
        self,
        skel_feat: torch.Tensor,
        video_feat: torch.Tensor,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            skel_feat:  (B, T, d_skel)
            video_feat: (B, T, d_video)
            causal:     是否加因果掩码
                True  → 第 t 帧只能看 ≤ t 的帧(流式,默认)
                False → 双向(离线推理用,精度可能略高)
        Returns:
            (fused_skel, fused_video)
                都是 (B, T, d_model)
        """
        s = self.skel_proj(skel_feat)      # (B, T, D)
        v = self.video_proj(video_feat)    # (B, T, D)

        nhead = self.nhead
        T = s.shape[1]

        # 因果掩码:帧级下三角(True = 禁止看未来)
        attn_mask = None
        if causal:
            from actionflow.models.streaming import build_causal_frame_mask

            # 传 num_joints=1 就退化成帧级掩码
            attn_mask = build_causal_frame_mask(T, 1, device=s.device)

        # ① 骨架查视频 → 语义补充
        semantic = self._attend(
            self.q_skel(s), self.k_video(v), self.v_video(v), nhead, attn_mask
        )
        semantic = self.out_skel(semantic)
        fused_skel = self.norm_skel(s + self.dropout(semantic))   # 残差连接

        # ② 视频查骨架 → 几何引导
        geometry = self._attend(
            self.q_video(v), self.k_skel(s), self.v_skel(s), nhead, attn_mask
        )
        geometry = self.out_video(geometry)
        fused_video = self.norm_video(v + self.dropout(geometry))  # 残差连接

        return fused_skel, fused_video


class FusionClassifier(nn.Module):
    """
    双路融合动作分类器。

    流程
    ----
        骨架 (B, T, J, 3)
          ↓ 骨架编码器
        骨架特征 (B, T, D)
          ┐
          ├─→ CrossAttentionFusion → 融合特征 (B, T, D)
          ┘
          ↓
        分类头 → (B, T, C)

        视频 (B, T, 3, H, W)
          ↓ VideoBackbone
        视频特征 (B, T, D_v)  ──→ CrossAttentionFusion

    关键:融合之后还会再过一层 Transformer,让模型能同时利用
    「当前帧的信息」和「前后帧的关系」。
    """

    def __init__(
        self,
        base_model,
        video_backbone: VideoBackbone | None = None,
        d_video: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
        use_video: bool = True,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.use_video = use_video
        self.d_model = base_model.d_model

        self.video_backbone = video_backbone
        self.fusion = CrossAttentionFusion(
            d_skel=self.d_model,
            d_video=d_video,
            d_model=self.d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # 融合后再过一层 Transformer(让融合结果能做时序推理)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.post_fusion_encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )

    # ------------------------------------------------------------------
    def _causal_mask(self, T: int, device: torch.device | None = None) -> torch.Tensor:
        """
        帧级因果掩码 (T, T)。

        ⚠️ **这一行不能省。**

        post_fusion_encoder 是一个**普通**的 TransformerEncoder,
        它默认每个 token 都能看到所有 token —— 包括未来的帧。

        如果不加因果掩码,第 3 帧的输出会依赖第 8 帧,
        这就**破坏了前缀不变性**,流式推理直接失效
        (实时系统里,已经输出的历史结果会被未来帧偷偷改变)。

        底座模型用的是"帧内全连通 + 帧间因果"的掩码
        (因为它的 token 粒度是 T*J);
        而这里 post_fusion_encoder 的 token 粒度是**帧**,
        所以掩码是简单的 (T, T) 下三角。

        两者的区别值得记一下:
            token 粒度 = 帧   → (T, T) 掩码,行 i 只看 j ≤ i
            token 粒度 = 关节 → (T*J, T*J) 掩码,同帧全连通 + 帧间因果
        """
        from actionflow.models.streaming import build_causal_frame_mask

        # 复用同一个构造函数:传 num_joints=1 就退化成帧级掩码
        return build_causal_frame_mask(T, 1, device=device)

    # ------------------------------------------------------------------
    def _encode_frames(self, skel: torch.Tensor) -> torch.Tensor:
        """
        骨架 → 每帧特征 (B, T, D)。

        底座模型有两种可能:
          - BoundaryAwareClassifier:自带 encode_frames,直接用
          - CausalStreamingClassifier:自己算一遍
        """
        if hasattr(self.base, "encode_frames"):
            return self.base.encode_frames(skel)

        B, T, J, _ = skel.shape
        x = self.base.embedding(skel)                      # (B, T*J, D)
        attn_mask, pad_mask = self.base.build_masks(skel)
        out = self.base.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        out = out.view(B, T, J, self.d_model)
        return out.mean(dim=2)                            # (B, T, D)

    # ------------------------------------------------------------------
    def forward(
        self,
        skel: torch.Tensor,
        frames: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            skel:   (B, T, J, 3)
            frames: (B, T, 3, H, W) 视频帧,用视频路时必填
            causal: 是否加因果掩码
                True  → 流式(默认):历史结果永不改写
                False → 离线:双向注意力,精度可能略高
        Returns:
            (B, T, num_classes)
        """
        T = skel.shape[1]

        feat = self._encode_frames(skel)

        if self.use_video and frames is not None and self.video_backbone is not None:
            video_feat = self.video_backbone(frames)   # (B, T, D_v)
            # ⚠️ 交叉注意力也必须加因果掩码 ——
            #    否则第 3 帧会去看第 8 帧的视频特征,一样破坏前缀不变性。
            feat, _ = self.fusion(feat, video_feat, causal=causal)

        # 融合后过一层 Transformer。
        # ⚠️ 必须传因果掩码,否则会破坏前缀不变性(见 _causal_mask 的说明)
        if causal:
            mask = self._causal_mask(T, device=skel.device)
            feat = self.post_fusion_encoder(feat, mask=mask)
        else:
            feat = self.post_fusion_encoder(feat)

        return self.base.classifier_head(feat)
