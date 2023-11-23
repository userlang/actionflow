"""
时序解码 (Temporal Decoding)
=============================

把「每帧独立的、带噪声的分类概率」变成「符合时序逻辑的、干净的动作段」。

为什么需要这一层
----------------
Transformer 分类头对每一帧**各自**输出类别概率，但它不知道
"相邻帧的动作一定是连续的"这条先验。于是会产生三类错误：

  1. 段被劈碎  —— 中间一帧误判，一个连续动作被切成两段
  2. 孤立假阳性 —— 某帧置信度突增 0.8，输出一个 0.03s 的动作段
  3. 边界抖动  —— 同一动作两次推理起止帧差 3~5 帧，时间戳不稳定

本模块提供三种互补的修复手段：

  ① Hysteresis 双阈值  → 治「段被切断」「边界抖动」
  ② 最小持续时长约束  → 治「孤立假阳性」
  ③ Viterbi 动态规划   → 治「局部最优」（全局一致性）

这三者叠加，输出才干净。详见 docs/technical-plan.html 第 15 节。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class ActionSegment:
    """一个动作段。"""

    start_frame: int
    end_frame: int          # 右开区间 [start, end)
    label_id: int
    confidence: float
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self, labels: list[str] | None = None) -> dict:
        name = labels[self.label_id] if labels and self.label_id < len(labels) else f"action_{self.label_id}"
        return {
            "start": round(self.start_time, 3),
            "end": round(self.end_time, 3),
            "duration": round(self.duration, 3),
            "label": name,
            "label_id": self.label_id,
            "confidence": round(self.confidence, 4),
        }


# ===========================================================================
# ① Hysteresis 双阈值（迟滞回环）
# ===========================================================================
def hysteresis_threshold(
    probs: np.ndarray,
    high: float = 0.7,
    low: float = 0.4,
) -> list[tuple[int, int, int]]:
    """
    用高/低双阈值把逐帧概率切成动作段。

    迟滞回环思想（与 Canny 边缘检测同源）：
      - 置信度 **涨到 high** 才认为动作**开始**
      - 置信度 **掉到 low 以下** 才认为动作**结束**
    这样动作中间的一两次低置信度误判不会把一个完整动作切成两半。

    段的结束有两种触发方式：
      a) 当前帧的**类别变了**（argmax 指向了另一个动作）→ 立即结束，开新段
      b) 当前帧置信度**跌破 low** → 结束

    Args:
        probs: (T, C) 每帧每类的概率
        high:  开启段的阈值
        low:   结束段的阈值（应 < high）
    Returns:
        [(start_frame, end_frame, label_id), ...]  end 为开区间
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs 应为 (T, C)，收到 {probs.shape}")
    if low > high:
        raise ValueError(f"low({low}) 不应大于 high({high})")

    best = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    segments: list[tuple[int, int, int]] = []
    active_cls = -1
    start = 0

    for t in range(len(best)):
        cur_cls = int(best[t])
        cur_conf = float(conf[t])

        if active_cls == -1:
            # 无活动段：需要高置信度才能开启
            if cur_conf >= high:
                active_cls = cur_cls
                start = t
            continue

        # ── 段进行中 ──
        if cur_cls == active_cls:
            # 同类别但置信度塌了 → 结束
            if cur_conf < low:
                segments.append((start, t, active_cls))
                active_cls = -1
            continue

        # ── 类别变了 ──
        # 只有当新类别**自信地占优**时才算真正的动作切换。
        # 若新类别的置信度本身很低（< high），说明这一帧是模糊的、
        # 不可靠的 —— 此时**跳过这一帧**，保持当前段继续，
        # 这正是迟滞回环要达到的效果。
        if cur_conf >= high:
            # 真正的切换：结束旧段，开启新段
            segments.append((start, t, active_cls))
            active_cls = cur_cls
            start = t
        elif cur_conf < low:
            # 模糊且低置信度：原类别也几乎不认为了 → 结束当前段
            if probs[t, active_cls] < low:
                segments.append((start, t, active_cls))
                active_cls = -1

    # 视频结束时仍在进行的段
    if active_cls != -1:
        segments.append((start, len(best), active_cls))

    return segments


# ===========================================================================
# ② Viterbi 动态规划（隐马尔可夫解码）
# ===========================================================================
def build_transition_matrix(
    num_classes: int,
    stay_prob: float = 0.9,
    switch_prob: float = 0.05,
) -> np.ndarray:
    """
    构造转移矩阵（对数概率）。

    设计意图：
        同动作 → 同动作   得分高  → 鼓励一段动作保持连续
        动作A → 动作B    得分中  → 允许切换，但有代价
        其余              得分低  → 允许从静止自由进出

    Args:
        num_classes:  类别数
        stay_prob:    对角线（保持）概率
        switch_prob:  非对角线（切换）基准概率
    Returns:
        (C, C) 对数概率矩阵（用 np.log 转过，数值更稳定）
    """
    A = np.full((num_classes, num_classes), switch_prob, dtype=np.float64)
    np.fill_diagonal(A, stay_prob)
    # 每行归一化，使转移成为合法概率分布
    A /= A.sum(axis=1, keepdims=True)
    return np.log(A)


def viterbi_decode(
    log_probs: np.ndarray,
    trans_logp: np.ndarray,
    start_logp: np.ndarray | None = None,
) -> np.ndarray:
    """
    Viterbi 解码：在所有可能的类别序列中找总得分最高的一条。

    目标:  argmax_y  Σ_t log P(y_t | y_{t-1}, x_t)
              ↑观测得分        ↑转移得分

    递推:  delta[t][s] = logP(s|x_t) + max_k( delta[t-1][k] + logA[k][s] )
                            ↑本帧观测    ↑从所有可能前驱转来的最优

    Args:
        log_probs:   (T, C) 每帧每类的对数概率
        trans_logp:  (C, C) 转移矩阵的对数概率
        start_logp:  (C,)   起始状态对数概率，默认均匀分布
    Returns:
        (T,) 每帧的最优类别 id
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, C = log_probs.shape

    if start_logp is None:
        start_logp = np.full(C, -np.log(C), dtype=np.float64)

    delta = np.full((T, C), -np.inf, dtype=np.float64)
    psi = np.zeros((T, C), dtype=np.int64)  # 记录最优前驱，用于回溯

    delta[0] = start_logp + log_probs[0]

    for t in range(1, T):
        # scores[k, s] = delta[t-1][k] + trans[k][s]
        scores = delta[t - 1][:, None] + trans_logp
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_probs[t]

    # 回溯
    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = int(delta[T - 1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]

    return path


def viterbi_smooth_probs(
    probs: np.ndarray,
    trans_logp: np.ndarray,
    blend: float = 0.5,
) -> np.ndarray:
    """
    用 Viterbi 路径平滑概率序列。

    关键点：**必须保留置信度的量级**。
    朴素做法是把 one-hot 路径和原概率按权重相加 —— 但那样会把高置信度
    帧的概率压得很低（例如 0.9 被拉成 0.55），导致后面的 Hysteresis
    双阈值失效。

    正确做法：对 Viterbi 选中的类别，**保留该帧原本的概率值**；
    只把其他类别的概率**按比例缩小**。这样：
      - 路径被 Viterbi 修正了（治「孤立误判」）
      - 置信度的大小关系被保留（治「阈值失效」）

    Args:
        probs:      (T, C) 概率
        trans_logp: (C, C) 转移矩阵对数概率
        blend:      0 = 完全不改（纯原概率），1 = 完全按 one-hot 覆盖
    Returns:
        (T, C) 平滑后的概率
    """
    probs = np.asarray(probs, dtype=np.float64)
    log_probs = np.log(np.clip(probs, 1e-12, None))
    path = viterbi_decode(log_probs, trans_logp)

    out = probs.copy()
    T, C = probs.shape
    for t in range(T):
        s = path[t]
        # 非选中类别的概率按 (1-blend) 缩放
        others = out[t].copy()
        others[s] = 0.0
        others *= 1.0 - blend
        out[t] = others
        # 选中类别保留原概率（若 blend>0 则向 1 靠拢）
        out[t, s] = probs[t, s] * (1.0 - blend) + blend * probs[t, s]
        # 重新归一化，保证仍是合法概率分布
        total = out[t].sum()
        if total > 1e-12:
            out[t] /= total

    return out


# ===========================================================================
# ③ 主解码入口
# ===========================================================================
def decode_segments(
    probs: np.ndarray,
    fps: float = 30.0,
    high: float = 0.7,
    low: float = 0.4,
    min_duration: float = 0.3,
    use_viterbi: bool = True,
    stay_prob: float = 0.9,
    switch_prob: float = 0.05,
    merge_gap: float = 0.0,
) -> list[ActionSegment]:
    """
    完整的时序解码流程。

    流程：
        1. (可选) Viterbi 平滑 —— 消除孤立误判
        2. Hysteresis 双阈值 —— 切出候选段
        3. 最小持续时长约束 —— 滤除假阳性
        4. 相邻同类段合并 —— 消除不必要的碎片
        5. 转换为带时间戳的 ActionSegment

    Args:
        probs:        (T, C) 每帧每类的概率
        fps:          帧率，用于时间戳换算
        high/low:     Hysteresis 双阈值
        min_duration: 最短动作时长（秒）
        use_viterbi:  是否启用 Viterbi 平滑
        stay_prob/switch_prob:  Viterbi 转移矩阵参数
        merge_gap:    间隔小于此秒数的相邻同类段会被合并（0 = 不合并）
    Returns:
        ActionSegment 列表
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs 应为 (T, C)，收到 {probs.shape}")
    if probs.size == 0:
        return []

    num_classes = probs.shape[1]

    # ── 步骤 1: Viterbi 平滑 ──────────────────────────────────────────
    if use_viterbi:
        trans_logp = build_transition_matrix(num_classes, stay_prob, switch_prob)
        probs = viterbi_smooth_probs(probs, trans_logp)

    # ── 步骤 2: Hysteresis 切段 ───────────────────────────────────────
    raw_segments = hysteresis_threshold(probs, high=high, low=low)

    # ── 步骤 3: 最小持续时长约束 ──────────────────────────────────────
    min_frames = max(1, int(round(min_duration * fps)))
    kept = [(s, e, c) for s, e, c in raw_segments if (e - s) >= min_frames]

    # ── 步骤 4: 合并相邻同类段 ────────────────────────────────────────
    if merge_gap > 0 and kept:
        gap_frames = int(round(merge_gap * fps))
        merged: list[tuple[int, int, int]] = [kept[0]]
        for s, e, c in kept[1:]:
            ps, pe, pc = merged[-1]
            if c == pc and s - pe <= gap_frames:
                merged[-1] = (ps, e, pc)
            else:
                merged.append((s, e, c))
        kept = merged

    # ── 步骤 5: 转换为带时间戳的动作段 ───────────────────────────────
    segments: list[ActionSegment] = []
    for s, e, c in kept:
        seg = probs[s:e, c]
        segments.append(
            ActionSegment(
                start_frame=int(s),
                end_frame=int(e),
                label_id=int(c),
                confidence=float(seg.mean()) if seg.size else 0.0,
                start_time=s / fps,
                end_time=e / fps,
            )
        )

    return segments


def format_segments(
    segments: list[ActionSegment],
    labels: list[str] | None = None,
) -> str:
    """把动作段格式化成人类可读的一行行文本。"""
    lines = []
    for seg in segments:
        name = labels[seg.label_id] if labels and seg.label_id < len(labels) else f"action_{seg.label_id}"
        lines.append(
            f"[{seg.start_time:.2f}-{seg.end_time:.2f}s] {name} (conf={seg.confidence:.3f})"
        )
    return "\n".join(lines)


# ===========================================================================
# ④ 边界回归修正
# ===========================================================================
def refine_boundaries_with_regression(
    segments: list[ActionSegment],
    boundary_dist: np.ndarray,
    fps: float = 30.0,
    max_adjust: int = 10,
    confidence: np.ndarray | None = None,
) -> list[ActionSegment]:
    """
    用边界回归的预测**微调** Hysteresis 切出来的段边界。

    问题
    ----
    Hysteresis 靠"置信度跨过阈值"来判定边界,天然偏晚:
    动作真正开始时模型往往还没把握(置信度只有 0.5),
    要等涨过 0.7 才敢开段 → 边界迟到了几帧。

    怎么修
    ------
    边界回归头对每一帧说:
        「我距**我所属那一段**的起点还有 d 帧,距终点还有 d' 帧」

    于是对段内任意一帧 t:
        所属段的起点位置 = t - bd[t, 0]
        所属段的终点位置 = t + bd[t, 1] + 1      (终点是开区间)

    这个式子在段内的每一帧上算出来都应该是**同一个数**
    ——因为它们描述的是同一个起点/终点。

    举例
    ----
        真值:抬手 = [30, 70),站立 = [70, 120)
        Hysteresis 迟到了,给出:抬手 = [32, 69)

        段内第 40 帧的回归预测:距起点 10,距终点 29
            起点 = 40 - 10 = 30  ✓
            终点 = 40 + 29 + 1 = 70  ✓
        → 修正后:抬手 = [30, 70)

    ⚠️ 为什么不能直接读段首帧?
    --------------------------
    段首帧是**歧义帧**。它可能是新段的第 1 帧(bd[.,0]=0),
    也可能是上一段的最后 1 帧(bd[.,0] 很大)。
    两种含义在数值上分不出来 —— 只有段**内部**的帧才不含歧义。

    所以本函数对段内帧做**中位数**统计(众数式的稳健估计),
    一两个异常值不会带偏结果。
    通俗讲:不看"门口那一帧",看"屋里的人怎么说的" ——
    屋里 10 个人都说「这段是从第 30 帧开始的」,那就信 30。

    安全措施
    --------
    1. **最多调整 max_adjust 帧** —— 回归头偶尔会给出离谱的值,
       不限制的话一个坏预测能把边界挪出去几十帧。
    2. **调整幅度不超过段长** —— 防止一个坏预测把整段吞掉/撑爆。
    3. **相邻段不重叠** —— 边界是"共享资源",冲突时统一处理。
    4. **不越出序列范围** —— 起点不能 < 0,终点不能 > T。

    Parameters
    ----------
    segments:      Hysteresis 切出来的段
    boundary_dist: (T, 2) 每帧到所属段起点/终点的距离(帧数)
    fps:           帧率
    max_adjust:    最多调整多少帧
    confidence:    (T, C) 概率表,用于重新计算段的置信度(可选)

    Returns
    -------
    修正后的 ActionSegment 列表
    """
    if not segments:
        return []

    boundary_dist = np.asarray(boundary_dist, dtype=np.float64)
    if boundary_dist.ndim != 2 or boundary_dist.shape[1] != 2:
        raise ValueError(f"boundary_dist 应为 (T, 2)，收到 {boundary_dist.shape}")

    T = boundary_dist.shape[0]

    # ══════════════════════════════════════════════════════════════
    # 第 1 遍:算出每一段"想要的"边界(先不管相邻段冲不冲突)
    # ══════════════════════════════════════════════════════════════
    #
    # ⚠️ 这里有个很容易踩的坑,值得写清楚。
    #
    # 回归头给每一帧报的是「距**它自己所属段**的起点/终点的距离」。
    # 看起来只要读段首帧的 bd[s,0] 就知道该往回挪多少,但**不行**。
    #
    # 为什么?
    #   假设真值是 抬手=[30,70),站立=[70,120)。
    #   Hysteresis 因为"要等置信度涨过阈值",把边界切在了 69:
    #       段2(抬手)=[32, 69)   段3(站立)=[69, 120)
    #   注意第 69 帧被分给了"站立",但**真值里它属于抬手**。
    #
    #   现在看第 69 帧的回归预测:
    #       bd[69,0] = 69 - 30 = 39   (它距抬手段的起点有 39 帧)
    #   如果直接信它,就会得出"站立段的起点要往前挪 39 帧" ——
    #   于是站立段被拽到第 30 帧,把整个抬手段压没了。
    #
    #   教训:**边界帧是"歧义帧"**。
    #   它既可能是新段的第 1 帧(bd[.,0]=0),
    #   也可能是上一段的最后 1 帧(bd[.,0] 很大)。
    #   两种含义在数值上无法区分 —— 只有看段**内部**的帧才能确定。
    #
    # 正确做法:对段内的帧,读出「所属段的起点位置」
    #       起点位置 = t - bd[t,0]
    #   这个值对段内的帧应该是**同一个数**(都在描述同一个起点)。
    #   段首帧是歧义帧,可能报出别的段的信息 → 用**中位数**取众数,
    #   一两个异常值不会影响结果。
    #
    #   通俗讲:不看"门口那一帧",看"屋里的人怎么说的" ——
    #   屋里 10 个人都说「这段是从第 30 帧开始的」,那就信 30。
    # ══════════════════════════════════════════════════════════════

    def _estimate_start(seg) -> int | None:
        """用段内帧的「所属段起点位置」的中位数来估计真实起点。"""
        s, e = seg.start_frame, seg.end_frame
        # 取段内的帧。首帧是歧义帧,排除掉;尾帧之前也要留一点余量。
        lo = min(s + 1, e - 1)
        hi = e
        if hi <= lo:
            return None
        idx = np.arange(lo, hi)
        idx = idx[idx < T]
        if idx.size == 0:
            return None
        est = idx - boundary_dist[idx, 0]
        # 中位数 = 众数式的稳健估计
        return int(round(float(np.median(est))))

    def _estimate_end(seg) -> int | None:
        """用段内帧的「所属段终点位置(开区间)」的中位数来估计真实终点。"""
        s, e = seg.start_frame, seg.end_frame
        # 末帧可能有歧义(它可能是下一段的首帧),排除掉
        lo, hi = s + 1, e - 1
        if hi <= lo:
            return None
        idx = np.arange(lo, hi)
        idx = idx[idx < T]
        if idx.size == 0:
            return None
        # 终点位置(开区间)= t + bd[t,1] + 1
        est = idx + boundary_dist[idx, 1] + 1.0
        return int(round(float(np.median(est))))

    def _has_interior_signal(seg, channel: int) -> bool:
        """
        段**内部**的帧有没有提供有效信息?

        为什么要判断?
        ------------
        回归预测有两种形态:
          ① 稠密型:段内每一帧都有"距所属段边界"的距离(训练时的标准形态)
          ② 稀疏型:只有边界帧上有值(比如人工构造的测试数据,
             或者只保留了边界帧的预测)

        对①,段内帧才是可靠信息源(边界帧有歧义) → 用中位数估计。
        对②,段内全是 0,中位数毫无意义 → 必须退回读边界帧。

        「内部」怎么取?
        ----------------
            起点用 [s+1, e)     —— 排除首帧(首帧可能是上一段的末帧)
            终点用 [s+1, e-1)   —— 首尾都排除(末帧可能是下一段的首帧)

        两种通道各自排除不同的帧,因为歧义的方向不一样:
            起点 → 首帧有歧义
            终点 → 末帧有歧义
        """
        s, e = seg.start_frame, seg.end_frame
        if channel == 0:
            lo, hi = s + 1, e
        else:
            lo, hi = s + 1, e - 1
        if hi <= lo:
            return False
        idx = np.arange(lo, hi)
        idx = idx[idx < T]
        if idx.size == 0:
            return False
        return bool(np.any(boundary_dist[idx, channel] > 0.5))

    def _fallback_start(seg) -> int:
        """
        兜底:段内帧没提供信息时,退回读段首帧。

        这对应另一种(更稀疏的)预测形态:只有边界帧上有值。
        边界帧读法:bd[s,0] = "这一帧迟到了多少帧才到起点" → 起点 = s - bd[s,0]
        """
        s = seg.start_frame
        if not (0 <= s < T):
            return s
        d = int(round(boundary_dist[s, 0]))
        return max(0, s - max(0, d))

    def _fallback_end(seg) -> int:
        """兜底:段尾前一帧的读法(见 _fallback_start 的说明)。"""
        e = seg.end_frame
        if not (0 <= e - 1 < T):
            return e
        d = int(round(boundary_dist[e - 1, 1]))
        return min(T, e + max(0, d))

    proposed: list[tuple[int, int]] = []   # (新起点, 新终点)
    for seg in segments:
        s, e = seg.start_frame, seg.end_frame

        # ── 起点修正 ───────────────────────────────────────────
        # 稠密预测:起点位置 = t - bd[t,0],取段内中位数 → 双向可信
        # 稀疏预测:只能读段首帧 → 语义是"迟到了多少帧" → 只能往回挪
        if _has_interior_signal(seg, 0):
            est_s = _estimate_start(seg)
            if est_s is None:
                est_s = _fallback_start(seg)
                shift = max(0, min(s - est_s, max_adjust))
            else:
                shift = s - est_s
                # 允许双向修正:
                #     est_s < s  → 边界迟到了,往回挪(这是边界回归头的主要价值)
                #     est_s > s  → 边界开早了,往后推
                # 幅度都被 max_adjust 卡住,防止一个坏预测把边界甩出去很远。
                shift = max(-max_adjust, min(max_adjust, shift))
        else:
            est_s = _fallback_start(seg)
            shift = max(0, min(s - est_s, max_adjust))     # 只能往回
        # 挪动幅度不能超过段本身的长度 —— 否则一个离谱的预测
        # 能把这一段整个吞掉(或撑爆)。
        shift = max(-(e - s), min(e - s, shift))
        new_s = max(0, min(T, s - shift))

        # ── 终点修正:同理 ───────────────────────────────────────
        if _has_interior_signal(seg, 1):
            est_e = _estimate_end(seg)
            if est_e is None:
                est_e = _fallback_end(seg)
                shift = max(0, min(est_e - e, max_adjust))
            else:
                shift = est_e - e
                shift = max(-max_adjust, min(max_adjust, shift))
        else:
            est_e = _fallback_end(seg)
            shift = max(0, min(est_e - e, max_adjust))     # 只能往后
        shift = max(-(e - s), min(e - s, shift))
        new_e = max(0, min(T, e + shift))

        proposed.append((new_s, new_e))

    # ══════════════════════════════════════════════════════════════
    # 第 2 遍:解决相邻段之间的冲突
    #
    # ⚠️ 这里的规则很容易搞错,值得说清楚:
    #
    #   段 A: [10, 30)   段 B: [32, 69)
    #   B 想往回挪 5 帧 → 想要 [27, 69)
    #
    #   「B 不能越过 A 的终点」吗?
    #   —— 如果这么约束,B 就永远没法往回修正了,
    #      而"往回修正"恰恰是边界回归头存在的唯一理由。
    #
    #   那正确的是什么?
    #   —— 正确规则:**相邻两段的边界应当落在同一个位置上**。
    #      A 的终点和 B 的起点是**同一个瞬间**(动作切换的那一帧),
    #      它们本来就是同一个数,只是分别被存了两份。
    #
    #   所以:如果 B 往回挪了 5 帧,那 A 的终点也应该跟着往前挪 5 帧。
    #      也就是 **把边界当成共享资源**,谁挪了就大家都挪。
    #
    #   实现方式:从左往右扫,维护一个"当前边界位置"。
    #     - 如果第 i 段的新终点 > 第 i-1 段的新起点(有重叠),
    #       就把第 i 段的新起点推到第 i-1 段的新终点。
    #     - 反过来(第 i 段往回挪过头,压到了前一段),
    #       就把**前一段的终点**也往前拉,保证不重叠。
    #
    #   最后再做一次兜底:任何一段都不能变成空段或负长度。
    # ══════════════════════════════════════════════════════════════
    resolved: list[tuple[int, int]] = []
    for i, (new_s, new_e) in enumerate(proposed):
        if i == 0:
            resolved.append((new_s, new_e))
            continue

        prev_s, prev_e = resolved[-1]

        if new_s < prev_e:
            # 本段往回挪过头,压到了前一段 → 把前一段的终点也往前拉
            # 这样两段的边界重新对齐,谁也不压谁
            new_prev_e = new_s
            if new_prev_e > prev_s:      # 前一段不能被拉成空段
                resolved[-1] = (prev_s, new_prev_e)
                prev_e = new_prev_e
            else:
                # 前一段被压没了 → 本段从原起点开始
                new_s = prev_e

        if new_e < new_s:
            new_e = new_s

        resolved.append((new_s, new_e))

    # ══════════════════════════════════════════════════════════════
    # 第 3 遍:转成 ActionSegment,重算置信度
    # ══════════════════════════════════════════════════════════════
    out: list[ActionSegment] = []
    for seg, (new_s, new_e) in zip(segments, resolved):
        # 兜底:不能是空段
        if new_e <= new_s:
            new_s, new_e = seg.start_frame, seg.end_frame

        # ── 重新算置信度(用修正后区间的平均概率) ───────────────
        conf = seg.confidence
        if confidence is not None:
            seg_probs = confidence[new_s:new_e, seg.label_id]
            if seg_probs.size:
                conf = float(seg_probs.mean())

        out.append(
            ActionSegment(
                start_frame=int(new_s),
                end_frame=int(new_e),
                label_id=seg.label_id,
                confidence=conf,
                start_time=new_s / fps,
                end_time=new_e / fps,
            )
        )

    return out
