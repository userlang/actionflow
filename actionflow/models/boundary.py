"""
边界回归头 (Boundary Regression Head)
=======================================

解决什么问题
------------
现在 Layer 3 用 Hysteresis 双阈值切段,切出来的边界**不够准**。

举个真实的例子:
    真实情况:  抬手动作从第 30 帧开始,第 70 帧结束
    Hysteresis: 第 33 帧才开始,第 68 帧就结束了

差个 3~5 帧听起来不多,但在 30fps 下就是 0.1~0.17 秒的误差。
如果你的产品要输出「这个动作从 1.20 秒开始」,
用户会拿秒表对着看 —— 差 0.1 秒就可能被发现了。

为什么双阈值会不准
------------------
Hysteresis 的判定依据只有**一个数**:当前帧的分类置信度。

    置信度 ≥ high(0.7) → 认为动作开始
    置信度 < low(0.4)  → 认为动作结束

问题在于:动作真正开始的那一刻,模型往往是**不太确定**的。
因为姿态模型刚看到"手开始往上动",还没看清到底要干什么。
所以真正的动作起点,置信度可能只有 0.45~0.55 —— 达不到 high。

结果就是:**系统要等好几帧,等到置信度涨过 0.7,才敢说是动作开始**。
这就天然产生了一个"迟到的边界"。

边界回归头的思路
----------------
**换个问法。**

分类头问的是:「这一帧是哪个动作?」(分类问题)
边界头问的是:「**这一帧距离动作的起点还有几帧?**」(回归问题)

第二个问题更容易回答!
因为"距离边界还有几帧"是个**连续的数字**,模型可以做细粒度预测;
而"是哪个动作"只有几个离散选项,天然粗糙。

打个比方:
    分类头像是"看照片猜这是什么",答案只能是"人/车/狗";
    边界头像是"看照片猜它在画面左边还是右边",答案可以是"偏左 15% 位置"。

实现方式
--------
在分类头旁边**并联**一个回归头,共享 Transformer 的输出特征:

    Transformer 输出 (B, T, J, D)
        ↓ 帧内池化
    帧特征 (B, T, D)
        ├─→ 分类头 →  logits (B, T, C)     「是哪个动作」
        └─→ 回归头 →  距离 (B, T, 2)       「距起点几帧 / 距终点几帧」

回归头输出的两个数字用 **softplus** 保证非负:

    距离 = softplus(原始输出)

为什么要 softplus?
    因为"距离"不可能是负数。
    softplus(x) = log(1 + e^x) 是一个平滑的"只能正数"的函数,
    在 x 接近 0 时它约等于 x,所以小数值不会被压扁。

    对比 ReLU:ReLU 在 x<0 时梯度直接变成 0,模型学不动了。
    softplus 没有这个问题,所以更适合做回归。

怎么用这个回归结果
------------------
拿到每帧的"距起点距离"和"距终点距离"后,就可以**修正** Hysteresis 切出来的边界:

    第 t 帧说"我距离最近的起点还有 3 帧"
    → 说明真正的起点在 t-3 附近
    → 把 Hysteresis 给出的 start_frame 往回挪

代码见 actionflow/postprocess/decode.py 的 refine_boundaries_with_regression()。

训练损失
--------
回归头的损失用 **L1 Loss**(绝对值误差),不用 MSE。

为什么用 L1 而不是 MSE?
    MSE(均方误差)对大误差惩罚特别重 —— 差 10 帧的惩罚是差 1 帧的 100 倍,
    梯度会被少数离群点主导,训练不稳定。
    L1(绝对值误差)对所有误差一视同仁,梯度恒定,更稳。

    这就像:你要教小孩把球放进筐里,
    MSE 相当于"球偏得越远骂得越狠",L1 相当于"偏多少都只扣一分"。
    后者学得更平稳。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryRegressionHead(nn.Module):
    """
    边界回归头 —— 预测每帧到动作起止边界的距离。

    输出
    ----
    (B, T, 2),最后一维两个通道分别是:
        通道 0: 距离**最近起点**的帧数(≥ 0)
        通道 1: 距离**最近终点**的帧数(≥ 0)

    通俗理解
    --------
    对第 37 帧,模型可能输出 (5, 12),意思是:
    「最近的起点在 5 帧之前(即第 32 帧)」
    「最近的终点在 12 帧之后(即第 49 帧)」

    于是这一帧就属于一个 [32, 49) 的动作段。
    把所有帧的预测组合起来,就能得到很精细的边界。

    参数
    ----
    d_model : int
        输入特征维度
    hidden : int
        隐藏层维度
    dropout : float
        dropout 比例
    """

    def __init__(self, d_model: int = 128, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, T, D) 每帧的特征
        Returns:
            (B, T, 2) 到起止边界的距离(帧数,非负)
        """
        raw = self.mlp(feat)
        # softplus 保证非负,且处处可导
        return nn.functional.softplus(raw)


class BoundaryAwareClassifier(nn.Module):
    """
    带边界回归头的动作分类器。

    结构
    ----
        骨架序列 (B, T, J, 3)
          ↓ 骨架编码器(与流式模型共用)
        帧特征 (B, T, D)
          ├─→ 分类头 → (B, T, C)  类别 logits
          └─→ 边界头 → (B, T, 2)  距离起止边界的帧数

    为什么要合并成一个模型?
    ----------------------
    因为分类和边界**共享同一份特征** —— 都靠 Transformer 理解"这个人在干嘛"。
    分成两个模型意味着要跑两遍 Transformer,慢一倍还可能给出矛盾的结论
    (分类说是"抬手",边界却说"这里没有边界")。

    合并之后,两个头看到的是同一份理解,天然一致。

    参数
    ----
    base_model : CausalStreamingClassifier
        底座模型(提供 embedding + encoder)
    use_boundary : bool
        是否启用边界头
    """

    def __init__(
        self,
        base_model,
        use_boundary: bool = True,
        boundary_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.use_boundary = use_boundary
        self.d_model = base_model.d_model
        self.num_classes = base_model.num_classes
        # 关节数从底座模型继承。
        # ⚠️ 必须暴露这个属性:ONNX 导出器要靠它造 dummy 输入
        #    (1, window, num_joints, 3)。底座模型有,包装层没有的话
        #    `export_streaming_onnx` 会报
        #    AttributeError: 'BoundaryAwareClassifier' object has no
        #    attribute 'num_joints' —— 也就是 --arch boundary 的模型
        #    根本导不出去。
        self.num_joints = base_model.num_joints

        # 边界头(可选)
        self.boundary_head = (
            BoundaryRegressionHead(self.d_model, hidden=boundary_hidden)
            if use_boundary
            else None
        )

    # ------------------------------------------------------------------
    def encode_frames(self, skel: torch.Tensor) -> torch.Tensor:
        """
        骨架 → 每帧特征 (B, T, D)。

        Parameters
        ----------
        skel : (B, T, J, 3)

        Returns
        -------
        (B, T, D) 每帧的 D 维特征
        """
        B, T, J, _ = skel.shape
        D = self.d_model

        x = self.base.embedding(skel)                      # (B, T*J, D)
        attn_mask, pad_mask = self.base.build_masks(skel)

        out = self.base.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        out = out.view(B, T, J, D)
        return out.mean(dim=2)                            # (B, T, D) 帧内均值池化

    # ------------------------------------------------------------------
    def forward(
        self,
        skel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            skel: (B, T, J, 3)
        Returns:
            (logits, boundary_dist)
                logits:       (B, T, C)
                boundary_dist: (B, T, 2) 或 None(未启用边界头时)
        """
        feat = self.encode_frames(skel)
        logits = self.base.classifier_head(feat)

        boundary = None
        if self.boundary_head is not None:
            boundary = self.boundary_head(feat)

        return logits, boundary

    # ------------------------------------------------------------------
    def forward_probs(
        self,
        skel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """同 forward,但分类输出走 softmax。"""
        logits, boundary = self.forward(skel)
        return logits.softmax(dim=-1), boundary


# ===========================================================================
# 训练损失
# ===========================================================================
def boundary_regression_loss(
    pred_dist: torch.Tensor,
    target_dist: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    边界回归损失(L1)。

    ⚠️ 量级说明
    -----------
    原始 L1(不做归一化)的量级是 **~15**(因为目标是帧数,一个 32 帧窗口
    里平均距离 15 帧),而分类的交叉熵只有 **~1.1**。
    直接相加 = 边界项压倒分类项,分类准确率会卡在随机水平。

    所以这里除以窗口长度,把损失归一化到 O(1) 量级,
    再由调用方用 ``boundary_weight`` 决定两者的权衡。

    Parameters
    ----------
    pred_dist : (B, T, 2)
        预测的距离
    target_dist : (B, T, 2)
        真值的距离
    valid_mask : (B, T) | None
        True 表示这一帧有效(参与了损失计算)。
        用于屏蔽"不属于任何动作段"的帧 —— 那些帧没有"最近边界"可言。

    Returns
    -------
    torch.Tensor 标量损失
    """
    if valid_mask is None:
        return nn.functional.l1_loss(pred_dist, target_dist)

    mask = valid_mask.unsqueeze(-1).to(pred_dist.dtype)   # (B, T, 1)
    err = (pred_dist - target_dist).abs()
    # ⚠️ 这里的归一化很重要。
    #
    # 回归目标是「距离(帧数)」,一个 32 帧窗口里平均距离是 ~15 帧,
    # 所以 L1 损失的天然量级是 **~15**,而分类的交叉熵只有 **~1.1**。
    # 直接相加的话,边界项会把分类项压掉 10 倍以上 ——
    # 实测表现: 训练 25 个 epoch 后分类准确率仍然卡在 33%(随机水平),
    # 模型只顾着优化回归,分类头根本没学到东西。
    #
    # 除以窗口长度把两个损失拉到同一个量级(都是 O(1)),
    # 再用 boundary_weight 明确控制两者的权衡。
    denom = mask.sum().clamp(min=1.0)
    return (err * mask).sum() / denom / max(mask.shape[1], 1)


def compute_boundary_targets(
    labels: torch.Tensor,
    num_classes: int,
    segment_bounds: list[list[tuple[int, int]]] | None = None,
    window_start: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据逐帧标签算出「每帧到最近起止边界的距离」。

    这是边界回归任务的**标签构造**。

    ⚠️ 关键:距离必须相对于**动作段边界**,而不是窗口边界
    -------------------------------------------------------
    推理时 ``postprocess/decode.py::refine_boundaries_with_regression``
    是这样反算边界的::

        起点 = t - dist_to_start[t]
        终点 = t + dist_to_end[t] + 1

    这里的 t 是**视频里的绝对帧号**。所以训练时的目标
    也必须是「到动作段边界的绝对距离」,两者语义才一致。

    如果只用窗口内的标签来算(旧做法):
        窗口是视频第 100~131 帧,整段都是类别 A
        → 算出的 dist_to_start = 0,1,2,...,31
        → 意思是「到**窗口**起点的距离」

    而推理时代码以为它是「到**动作段**起点的距离」——
    训练/推理语义不一致,边界修正必然把边界挪到错误位置。

    参数
    ----
    labels : (B, T)
        逐帧类别 id。
    num_classes : int
        类别总数。
    segment_bounds : list[list[tuple[int, int]]] | None
        每个样本的**绝对**动作段边界 ``[(start, end), ...]``
        (相对整段视频的绝对帧号,end 是开区间)。
        给出时用它算目标 —— 这才是与推理一致的语义。

        为 None 时退回到「按窗口内标签变化推断边界」,
        得到的是窗口相对距离,只适用于不接入 decode.py 的离线评估。

    window_start : list[int] | None
        每个样本窗口在视频中的**绝对起始帧号**。
        给了它才能把绝对段边界换算成窗口内相对坐标。

    Returns
    -------
    (dist_to_start, dist_to_end)
        都是 (B, T) 的 float 张量
    """
    B, T = labels.shape
    device = labels.device

    dist_to_start = torch.zeros(B, T, dtype=torch.float32, device=device)
    dist_to_end = torch.zeros(B, T, dtype=torch.float32, device=device)

    # ── 优先:用绝对段边界(与推理语义一致) ────────────────────
    if segment_bounds is not None:
        for b in range(B):
            bounds = segment_bounds[b] if b < len(segment_bounds) else []
            # bounds 里是**绝对**帧号,窗口起点是 window_start[b]
            # → 先转成窗口内相对坐标,再算距离。
            # 距离本身是「到段边界的帧数」,与坐标系无关。
            off = window_start[b] if window_start is not None and b < len(window_start) else 0
            for t in range(T):
                abs_t = t + off
                # 找这一帧所属的段(取包含它的第一个段)
                for s, e in bounds:
                    if s <= abs_t < e:
                        dist_to_start[b, t] = float(abs_t - s)
                        dist_to_end[b, t] = float(e - abs_t)
                        break
                else:
                    # 不属于任何段(标签无效)→ 兜底
                    dist_to_start[b, t] = float(t)
                    dist_to_end[b, t] = float(T - 1 - t)
        return dist_to_start, dist_to_end

    # ── 兜底:按窗口内标签变化推断边界 ──────────────────────────
    for b in range(B):
        row = labels[b]
        # 前向:距最近起点
        for t in range(T):
            if t == 0 or row[t] != row[t - 1]:
                dist_to_start[b, t] = 0.0
            else:
                dist_to_start[b, t] = dist_to_start[b, t - 1] + 1.0

        # 后向:距最近终点
        for t in range(T - 1, -1, -1):
            if t == T - 1 or row[t] != row[t + 1]:
                dist_to_end[b, t] = 0.0
            else:
                dist_to_end[b, t] = dist_to_end[b, t + 1] + 1.0

    return dist_to_start, dist_to_end


def build_boundary_aware_model(cfg) -> BoundaryAwareClassifier:
    """从 ModelConfig 构建带边界头的模型。"""
    from actionflow.models.streaming import CausalStreamingClassifier

    base = CausalStreamingClassifier(
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
    return BoundaryAwareClassifier(base, use_boundary=True)
