"""
解码层单元测试 —— 验证三种病症各自被对应的手段治好。

运行:  pytest tests/test_decode.py -v
或:    python -m tests.test_decode
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.postprocess.decode import (  # noqa: E402
    build_transition_matrix,
    decode_segments,
    hysteresis_threshold,
    viterbi_decode,
)

FPS = 30.0
C = 4


def make_probs() -> np.ndarray:
    """构造一段带噪声的动作序列：0-1s 动作0, 1-2s 动作1, 2-3s 动作2, 3-4s 动作3"""
    T = 120
    probs = np.zeros((T, C), dtype=np.float64)
    truth = [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30

    rng = np.random.default_rng(0)
    for t, c in enumerate(truth):
        probs[t, c] = float(np.clip(0.9 + rng.normal(0, 0.05), 0, 1))
        rest = (1.0 - probs[t, c]) / (C - 1)
        for k in range(C):
            if k != c:
                probs[t, k] = rest * (0.5 + rng.random())
    return probs


def test_naive_argmax_is_fragmented() -> None:
    """病症 1: 朴素 argmax 在有噪声时产生大量碎片。"""
    probs = make_probs()
    # 注入一个孤立误判，模拟真实模型输出
    probs[95, :] = 0.05
    probs[95, 1] = 0.85

    naive = probs.argmax(1)
    runs = 1 + int((naive[1:] != naive[:-1]).sum())
    assert runs >= 5, f"朴素 argmax 应该碎片很多，实际 {runs} 段"


def test_viterbi_removes_isolated_misprediction() -> None:
    """病症 2: Viterbi 把孤立误判拉回去。"""
    probs = make_probs()
    probs[35, :] = 0.05
    probs[35, 3] = 0.90  # 孤立一帧误判

    trans = build_transition_matrix(C, stay_prob=0.9, switch_prob=0.05)
    path = viterbi_decode(np.log(np.clip(probs, 1e-12, None)), trans)

    # 第 35 帧应被拉回动作1（真值）
    assert path[35] == 1, f"Viterbi 应该拉回第35帧，实际 {path[35]}"
    # 路径应该连续（切换次数很少）
    switches = int((path[1:] != path[:-1]).sum())
    assert switches <= 4, f"路径切换次数应该很少，实际 {switches}"


def test_hysteresis_survives_confidence_drop() -> None:
    """
    病症 3: Hysteresis 在置信度短暂下降时不切断动作段。

    场景：动作1 进行到一半，某一帧所有类别都不确定，但**原类别仍是最高分**
    （argmax 没变，只是置信度降到了 0.45）。此时不应切段。
    """
    probs = make_probs()
    # 把 45 帧压成「都不确定，但动作1 仍最高」的情况
    probs[45, :] = [0.15, 0.45, 0.25, 0.15]

    segs = hysteresis_threshold(probs, high=0.7, low=0.4)
    seg_1 = [(s, e, c) for s, e, c in segs if c == 1]
    assert len(seg_1) == 1, f"动作1 应该只有一段，实际 {seg_1}"
    s, e, _ = seg_1[0]
    assert e - s >= 28, f"动作1 段长度应该接近 30 帧，实际 {e-s}"


def test_hysteresis_closes_segment_when_all_confidence_lost() -> None:
    """
    对照场景：如果某帧**所有**类别都不确定（置信度全面塌陷），
    说明这一帧没有任何有效信息，应当结束当前段（而不是硬撑下去）。
    """
    probs = make_probs()
    probs[45, :] = 0.05  # 全塌
    probs[45, 2] = 0.25  # argmax 翻到动作2，但置信度很低

    segs = hysteresis_threshold(probs, high=0.7, low=0.4)
    seg_1 = [(s, e, c) for s, e, c in segs if c == 1]
    assert len(seg_1) == 2, f"置信度全面塌陷时应该切段，实际 {seg_1}"


def test_hysteresis_switches_on_confident_new_class() -> None:
    """当新类别自信地占优（>= high）时，必须立即切换。"""
    probs = make_probs()
    probs[45, :] = 0.05
    probs[45, 2] = 0.85  # 动作2 高置信度抢占

    segs = hysteresis_threshold(probs, high=0.7, low=0.4)
    labels = [c for _, _, c in segs]
    assert 2 in labels, f"应该切到动作2，实际段标签 {labels}"


def test_min_duration_filters_false_positive() -> None:
    """病症 4: 0.03s 的孤立假阳性段被最小时长约束滤掉。"""
    probs = make_probs()
    probs[95, :] = 0.05
    probs[95, 1] = 0.85  # 孤立一帧高置信度

    # 不加最小时长约束时，会产生一个 1 帧的假段
    raw = hysteresis_threshold(probs, high=0.7, low=0.4)
    assert any(e - s == 1 for s, e, _ in raw), "应该存在 1 帧的假阳性段"

    # 加上最小时长约束后被滤除
    final = decode_segments(probs, fps=FPS, min_duration=0.3, use_viterbi=False)
    assert all(seg.num_frames >= 9 for seg in final), "不应该有短于 9 帧的段"


def test_full_decode_recovers_ground_truth() -> None:
    """端到端：完整解码能把 4 段还原出来。"""
    probs = make_probs()
    probs[35, :] = 0.05
    probs[35, 3] = 0.90
    probs[95, :] = 0.05
    probs[95, 1] = 0.85

    segs = decode_segments(
        probs, fps=FPS, high=0.7, low=0.4, min_duration=0.3, use_viterbi=True
    )
    labels = [s.label_id for s in segs]
    assert labels == [0, 1, 2, 3], f"应该还原出 4 个动作段，实际 {labels}"
    # 置信度应该保持在合理量级（没有被 one-hot 压低）
    assert all(s.confidence > 0.5 for s in segs), "置信度量级被破坏了"


def test_viterbi_preserves_confidence_scale() -> None:
    """Viterbi 平滑不能把高置信度压到阈值以下。"""
    from actionflow.postprocess.decode import viterbi_smooth_probs

    probs = make_probs()
    trans = build_transition_matrix(C, stay_prob=0.9, switch_prob=0.05)
    smoothed = viterbi_smooth_probs(probs, trans)

    # 平滑后，选中类别的概率应该仍然很高
    best = smoothed.max(axis=1)
    assert best.min() > 0.7, f"平滑后置信度不应低于 0.7，实际最低 {best.min():.3f}"


def test_transition_matrix_rows_sum_to_one() -> None:
    """转移矩阵每行必须归一化。"""
    trans = np.exp(build_transition_matrix(C, stay_prob=0.9, switch_prob=0.05))
    assert np.allclose(trans.sum(axis=1), 1.0), "转移矩阵行和应为 1"


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
            print(f"  ✗ {name}\n      {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name}\n      {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
