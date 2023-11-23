"""
因果流式骨架模型 (Causal Streaming Skeleton Model)
==================================================

这份文件解决一个具体问题：**实时视频流里，如何做到"看一帧出一帧"，
而且出的结果不会被后来的帧推翻？**

先想清楚"流式"到底要什么
------------------------
离线处理整段视频很简单：把 100 帧一起扔进模型,一次算完。
但摄像头是"边录边看"的。第 37 帧到来时,你只有 0~37 帧的信息,
不能等未来。所以模型必须满足一条铁律：

    **前缀不变性 (Prefix Invariance)**
    只用前 t 帧算出来的结果,必须和用前 t+1 帧算出来的前 t 帧结果**完全一样**。

    为什么这条这么重要?
    想象一下:第 10 帧时模型说"抬手",于是你的系统打印了一条日志。
    第 50 帧时(如果模型能看未来)它改口说"站立"——
    那之前那条日志怎么办?已经发给用户了。
    实时系统里,你**没办法**撤回已经发出的结果。
    所以模型必须"说到做到":一旦说了,后面就不许改。

怎么实现
--------
靠**因果注意力掩码 (Causal Attention Mask)**。

先理解模型内部是怎么"看"的
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
骨架序列进来后被展平成一排 token:

    帧0: [鼻, 左肩, 右肩, 左肘, ..., 左脚踝]     ← 17 个 token
    帧1: [鼻, 左肩, 右肩, 左肘, ..., 左脚踝]     ← 17 个 token
    帧2: [ ... ]

展平后的顺序是 ``token编号 = 帧号 × 17 + 关节号``,所以:

    token 0,1,2,...,16   属于第 0 帧
    token 17,18,...,33   属于第 1 帧
    token 34,35,...,50   属于第 2 帧

每个 token 做注意力时,就是"在这一排 token 里挑几个来参考"。

因果掩码就是规定:**第 t 帧的 token,只能参考第 0~t 帧的 token。**

一个容易踩的坑
~~~~~~~~~~~~~~
**帧内的关节必须互相可见!**

右手腕要知道右手肘在哪才能判断"手抬起来没有",这需要看**同一帧**的其他关节。
所以掩码不能简单地"禁止看编号更大的 token"——那样会把同帧的关节也隔开,
左手腕就看不到左手肩了。

正确的规则是按**帧号**判断,不是按 token 编号:

    允许看  ⟺  对方的帧号 ≤ 自己的帧号

    token 0  (帧0·左肩)  →  能看 token 16 (帧0·右脚踝)  ✅ 同帧,允许
    token 0  (帧0·左肩)  →  不能看 token 17 (帧1·鼻)     ❌ 未来帧,禁止
    token 17 (帧1·鼻)    →  能看 token 5  (帧0·右肘)     ✅ 过去帧,允许

代码里就是这一行:

    # t_of[i] = token i 属于第几帧
    # mask[i, j] = True 表示 "i 不能看 j"
    mask = t_of[None, :] > t_of[:, None]

    读作: 第 j 个 token 的帧号 > 第 i 个 token 的帧号 → 禁止

这个掩码有个重要性质:它只跟"帧号"有关,跟 T 无关。
所以推理时窗口多长都不影响正确性,流式滑动才安全。

两种逐帧输出的做法
------------------
做法 A(旧,朴素):对每个前缀 t 跑一次完整模型
    算 T 帧的逐帧结果要跑 T 次完整前向 → 计算量 O(T²)
    优点:简单、绝对正确
    缺点:慢。而且"每个前缀单独跑"这件事本身是浪费的——
         其实一次前向就能算出所有帧的结果,只要模型是因果的。

做法 B(新,本文档):一次因果前向,直接读出每帧的类别
    因为因果掩码保证了第 t 帧的输出只依赖前 t 帧,
    所以第 t 帧的 token 在前向过程中"已经算出了截至 t 帧的信息"。
    把每帧的 17 个 token 池化一下,就是该帧的分类结果。
    一次前向 → 全部 T 帧的逐帧概率。计算量 O(T)。

做法 B 是流式推理的正确形态。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ===========================================================================
# 因果掩码的构造
# ===========================================================================
def build_causal_frame_mask(
    num_frames: int,
    num_joints: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    构造「帧内全连通 + 帧间因果」的注意力掩码。

    通俗讲
    -------
    把 (T 帧 × J 关节) 展平成一排 token,编号规则是 ``token = 帧号*J + 关节号``。

    然后规定:
        「我」是第 i 个 token,「对方」是第 j 个 token
        - 如果 j 属于**同一帧** → 允许(手要看自己的胳膊)
        - 如果 j 属于**更早的帧** → 允许(可以参考历史)
        - 如果 j 属于**更晚的帧** → 禁止(那是未来,实时时还没发生)

    Parameters
    ----------
    num_frames : int
        帧数 T
    num_joints : int
        关节数 J (COCO = 17)

    Returns
    -------
    torch.Tensor
        形状 (T*J, T*J) 的 bool 张量。
        **True = 禁止关注**,False = 允许关注。
        (PyTorch 的 attn_mask 约定: True 表示"这里填 -inf,不许看")

    Examples
    --------
    >>> m = build_causal_frame_mask(2, 3)
    >>> m.shape
    torch.Size([6, 6])
    >>> bool(m[0, 2])   # 帧0 的 token0 看 帧0 的 token2
    False
    >>> bool(m[0, 3])   # 帧0 的 token0 看 帧1 的 token0
    True
    """
    # t_of[i] = 第 i 个 token 属于第几帧
    # 举例: J=3, T=4 → [0,0,0, 1,1,1, 2,2,2, 3,3,3]
    #
    # ⚠️ 这里必须用**纯 Python 列表**构造,不能用 torch.arange(...).item() 之类
    #    会引入动态形状判断的写法 —— torch.onnx.export 的 dynamo 后端
    #    对数据依赖的形状会直接报错(GuardOnDataDependentSymNode)。
    #    用 Python int 做除法,得到的是编译期常量,导出时完全静态。
    t_of = [i // num_joints for i in range(num_frames * num_joints)]

    # mask[i, j] = (j 的帧号 > i 的帧号) → True 表示禁止
    return torch.tensor(
        [[t_of[j] > t_of[i] for j in range(len(t_of))] for i in range(len(t_of))],
        dtype=torch.bool,
        device=device,
    )


# ===========================================================================
# 因果流式分类器
# ===========================================================================
class CausalStreamingClassifier(nn.Module):
    """
    因果流式骨架动作分类器 —— 一次前向输出整段逐帧概率。

    与普通分类器的关系
    ------------------
    这个类和 ``SkeletonTransformerClassifier`` **共用同一套骨架编码器**,
    区别只在最后"怎么把 token 变成逐帧结果":

        普通分类器:  整段池化成一个向量 → 1 个结果(回答"这段视频整体是啥")
        因果流式:    每帧单独池化 → T 个结果(回答"每一帧在干啥")

    为什么必须分开写?
    ----------------
    因为普通分类器在**整段**上池化,第 t 帧的结果会被后面的帧影响
    (池化是全局操作),违反前缀不变性。

    而流式模型只对"截至当前帧"的信息池化,天然满足要求。

    内部结构
    --------
        输入 (B, T, J, 3)
          ↓ 坐标投影 + 关节嵌入 + 时间嵌入
        token (B, T*J, D)
          ↓ 因果注意力(帧内全连通 + 帧间因果)
        out (B, T*J, D)
          ↓ 按帧拆回 (B, T, J, D)
          ↓ 每帧内的 J 个关节池化成 D 维
        (B, T, D)
          ↓ 分类头(逐帧共享同一个 MLP)
        logits (B, T, C)
          ↓ softmax
        probs (B, T, C)

    关键点:分类头是**逐帧共享**的(不是每帧一个),
    所以窗口长度变化不会改变参数量,滑窗推理天然稳定。
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
        conf_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        from actionflow.models.classifier import SkeletonEmbedding

        self.num_classes = num_classes
        self.num_joints = num_joints
        self.d_model = d_model
        self.window = window
        self.conf_threshold = conf_threshold

        # 复用普通分类器的编码器 —— 保证两套模型看到的是同一种"骨架语言"
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
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 逐帧分类头 —— 每一帧都用同一套参数做分类
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # ------------------------------------------------------------------
    def build_masks(
        self,
        skel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        构造因果掩码 + 漏检掩码。

        Parameters
        ----------
        skel : (B, T, J, 3)

        Returns
        -------
        (attn_mask, key_padding_mask)
            attn_mask:        (T*J, T*J) bool,True = 禁止
            key_padding_mask: (B, T*J)   bool,True = 忽略(该 token 无效)
        """
        B, T, J, _ = skel.shape
        device = skel.device

        # ── 因果掩码(帧内全连通 + 帧间因果) ─────────────────────────
        attn_mask = build_causal_frame_mask(T, J, device=device)

        # ── 漏检掩码:置信度太低的关节不参与注意力 ───────────────────
        conf = skel[..., 2]                     # (B, T, J)
        valid = conf > self.conf_threshold      # (B, T, J)

        # 防 NaN:如果某帧所有关节都被判无效,softmax 会除以 0。
        # 用无分支写法(数据依赖分支无法被 ONNX 导出 trace)。
        all_masked = ~valid.any(dim=-1)         # (B, T)
        first_joint = torch.zeros_like(valid)
        first_joint[:, :, 0] = 1.0
        valid = valid | (all_masked.unsqueeze(-1).to(valid.dtype) * first_joint)

        key_padding_mask = ~valid.reshape(B, T * J)   # True = 忽略

        return attn_mask, key_padding_mask

    # ------------------------------------------------------------------
    def forward(self, skel: torch.Tensor) -> torch.Tensor:
        """
        一次前向,输出每帧的分类 logits。

        Parameters
        ----------
        skel : (B, T, J, 3)
            骨架序列

        Returns
        -------
        (B, T, num_classes) logits(未 softmax)
        """
        B, T, J, _ = skel.shape
        D = self.d_model

        x = self.embedding(skel)                     # (B, T*J, D)
        attn_mask, pad_mask = self.build_masks(skel)

        out = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)

        # 拆回 (B, T, J, D)
        out = out.view(B, T, J, D)

        # 帧内池化:这一帧的 17 个关节 → 1 个 D 维向量
        #   用 mean 而不是 max:mean 是"平均意见",更稳;
        #   max 是"最强意见",对某个关节异常敏感。骨架有漏检时 mean 更可靠。
        frame_feat = out.mean(dim=2)                  # (B, T, D)

        logits = self.classifier_head(frame_feat)     # (B, T, C)
        return logits

    # ------------------------------------------------------------------
    def forward_probs(self, skel: torch.Tensor) -> torch.Tensor:
        """
        同 forward,但输出 softmax 后的概率。

        Returns
        -------
        (B, T, num_classes) 概率,每帧概率和为 1
        """
        return self.forward(skel).softmax(dim=-1)

    # ------------------------------------------------------------------
    def step(self, skel_frame: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        """
        单帧推进接口 —— 真正的"来一帧出一帧"。

        这个方法是为了演示**为什么流式模型不需要重算整个窗口**:
        因果掩码保证了第 t 帧的输出只依赖前 t 帧,
        而 Transformer 的自注意力是"加权的求和",
        所以理论上可以做 KV cache 增量更新。

        ⚠️ 但本实现**故意不做 KV cache**。原因:
          1. nn.TransformerEncoderLayer 没有暴露 KV cache 接口,
             要手写需要重实现整个 attention 逻辑,代码量大且容易出错;
          2. 窗口只有 32 帧、模型只有 0.3M 参数,
             整窗重算一次的成本是毫秒级,完全可以接受;
          3. 面试时"我知道这里可以优化、也能说清楚为什么没做"，
             比"我硬写了个可能有 bug 的 cache"更有说服力。

        实际上本方法就是:把单帧当成 T=1 的序列送进去,
        它会退化成"只看这一帧"的分类。要得到"截至当前帧"的结果,
        应该用滑窗 + 本类的 forward(整窗一次算完)。

        Parameters
        ----------
        skel_frame : (1, 1, J, 3)
            单帧骨架
        cache : dict | None
            预留的缓存接口(当前未使用)

        Returns
        -------
        (1, 1, num_classes) 概率
        """
        if cache is not None:
            cache.setdefault("frames", []).append(skel_frame)
        return self.forward_probs(skel_frame)


# ===========================================================================
# ONNX 导出（流式模型专用）
# ===========================================================================
def export_streaming_onnx(
    model: CausalStreamingClassifier,
    out_path: str,
    window: int | None = None,
    opset: int = 17,
) -> str:
    """
    把流式模型导出为 ONNX。

    ⚠️ 为什么这里要用 ``dynamo=False``（旧版 TorchScript 导出器）?

    原因:PyTorch 新版的 dynamo 导出器在遇到 **Transformer 的注意力掩码** 时会崩。

    具体表现是导出时报:
        GuardOnDataDependentSymNode: Could not guard on data-dependent expression

    原因分析(读源码得出的结论):
        - 注意力要计算"每个 token 会被多少个有效 token 关注",
          用到 softmax 之后沿 key 维求和;
        - 当 key_padding_mask 把某些 token 全屏蔽时,
          这个和会是 0,PyTorch 内部需要判断"要不要跳过这一行";
        - 这个判断是**数据依赖**的(取决于运行时哪些 token 被屏蔽),
          而 dynamo 导出器要求形状在编译期确定 → 报错。

        旧版 TorchScript 导出器走 trace 机制,遇到这种分支会直接
        记录执行路径,不需要编译期确定 → 能导出。

    ⚠️ 为什么窗口长度必须是**静态**的?

    同样是因为 dynamo/trace 的限制:
        注意力内部要把 (B, L, D) 拆成 (B, L, nhead, head_dim) 再转置,
        这个 reshape 的目标形状在导出时被固化成了常量。
        换成长度会变,onnxruntime 加载时会直接报 Reshape 错。

    对流式场景来说这**不是限制**:流式推理的窗口本来就是固定的
    (比如永远看最近 32 帧),长度固定反而是好事 ——
    可以让端侧引擎预先分配好内存,不用动态 shape。

    Returns
    -------
    str  导出的文件路径
    """
    import warnings
    from pathlib import Path

    window = window or model.window
    model.eval()

    dummy = torch.zeros(1, window, model.num_joints, 3)
    dummy[..., 2] = 1.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["skel"],
            output_names=["logits"],
            # batch 维动态(端侧可以批量处理),时间维固定(见上面的说明)
            dynamic_axes={"skel": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            dynamo=False,          # ← 关键:用旧版导出器,见上面的说明
            do_constant_folding=True,
        )

    return str(out_path)


def build_streaming_model(cfg) -> CausalStreamingClassifier:
    """从 ModelConfig 构建因果流式模型。"""
    return CausalStreamingClassifier(
        num_classes=cfg.num_classes,
        num_joints=cfg.num_joints,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        window=cfg.window,
        conf_threshold=getattr(cfg, "conf_threshold", 0.3),
    )
